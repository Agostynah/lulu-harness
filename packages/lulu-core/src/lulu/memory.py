"""memory.py: MemoryStore -- the harness's actual local memory, wired
into AgentLoop (loop.py). Without this, ContextAssembler and MemoryRouter
are tested, working components that nothing in a real `lulu` run ever
calls -- this is the piece that makes the thesis true of the running
harness, not just of evals/dbpedia.

Backed by lulu_router's InMemoryShardStore for v0. Honestly in-process
only -- memory does not survive a restart. A persistent TurboVec+SQLite
backend (ported from local-memory/core/vector_store.py, as originally
scoped) is the natural next step, tracked in ROADMAP.md, not built here:
the router and ContextAssembler don't care what backs a Shard, so
swapping the store later touches only this file.

Shards are partitioned by TYPE (episodic/semantic), not KMeans clustering.
KMeans is what evals/dbpedia exercises over a 100K-document research
corpus; a harness's day-to-day memory starts small and is already
naturally categorized by why it was written, so clustering it would be
solving a problem that doesn't exist yet at this scale.

Scope isolation, and a real bug caught by adversarial review rather than
assumed safe: an earlier version kept exactly ONE physical Shard per
type (e.g. one "episodic" Shard) and, on a scoped write, UNIONED the new
scope into that shard's allowed_scopes. That meant once two different
scopes both wrote to "episodic", shard.permits() passed for either one --
and since InMemoryShardStore has no per-vector scope of its own, the
WHOLE merged store (both scopes' content, physically concatenated by
write()) became searchable by both. This directly contradicted
evals/leakage.py's proven claim, which never actually exercised this path
(it builds separate shards by hand, bypassing write() entirely). Fixed by
keying shards on (type, scope) instead of type alone: each scope gets its
own physically separate Shard per type, and a search only ever sees the
shards for its own scope -- see shards_for_scope(). See
test_memory.py's cross-scope-on-shared-type regression tests.
"""

from __future__ import annotations

import numpy as np
from lulu_router.backends.memory import InMemoryShardStore
from lulu_router.cost import Budget, CostProfile
from lulu_router.judges.geometric import GeometricJudge
from lulu_router.router import MemoryRouter
from lulu_router.shard import Shard

from lulu.context import AssembledContext, ContextAssembler
from lulu.embeddings import Embedder

EPISODIC_COST = CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=80)
SEMANTIC_COST = CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=120)

DEFAULT_SHARD_COSTS: dict[str, CostProfile] = {
    "episodic": EPISODIC_COST,
    "semantic": SEMANTIC_COST,
}


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 1e-9 else vec


class MemoryStore:
    def __init__(
        self,
        embedder: Embedder | None = None,
        shard_costs: dict[str, CostProfile] | None = None,
        strategy: str = "progressive_expansion",
        k: int = 5,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.shard_costs = shard_costs or DEFAULT_SHARD_COSTS
        self.strategy = strategy
        self.k = k
        self.judge = GeometricJudge()
        # Keyed on (shard_type, scope) -- NOT just shard_type. This is
        # the actual fix: two different scopes writing to "episodic"
        # get two distinct Shard objects (distinct InMemoryShardStore
        # instances too), never one shared store with a unioned
        # permission set. See module docstring.
        self._shards: dict[tuple[str, str | None], Shard] = {}

    def _shard(self, shard_type: str, scope: str | None) -> Shard:
        if shard_type not in self.shard_costs:
            raise ValueError(f"unknown shard {shard_type!r}; known shards: {list(self.shard_costs)}")
        key = (shard_type, scope)
        if key not in self._shards:
            shard_id = shard_type if scope is None else f"{shard_type}:{scope}"
            self._shards[key] = Shard(
                id=shard_id,
                store=InMemoryShardStore(),
                cost=self.shard_costs[shard_type],
                allowed_scopes=None if scope is None else frozenset({scope}),
            )
        return self._shards[key]

    def shards_for_scope(self, scope: str | None) -> list[Shard]:
        """Every shard a caller with this scope may legitimately search --
        its own scope's shards, and only its own, regardless of how many
        other scopes happen to also have written to a shard of the same
        *type*. Used both by search() and by callers rendering /cost
        (cli.py, server.py), which must show a scope-appropriate
        counterfactual rather than one that includes other scopes'
        shards -- that would itself leak "how much data exists for other
        tenants," a subtler version of the same class of bug."""
        return [shard for (_shard_type, shard_scope), shard in self._shards.items() if shard_scope == scope]

    def write(self, content: str, shard: str = "episodic", scope: str | None = None) -> None:
        # O(n) per write, not O(1): InMemoryShardStore is immutable by
        # design (see backends/memory.py), so every write rebuilds the
        # whole shard -- copies every existing id/content and re-stacks
        # every existing vector -- to append one row. n writes to one
        # shard is therefore O(n^2) total. Fine at harness scale (a
        # session writes tens to low hundreds of memories); the honest fix
        # is a mutable/append-only backend (the TurboVec+SQLite backend
        # already tracked in this file's module docstring and
        # ROADMAP.md), not optimizing this loop.
        target = self._shard(shard, scope)
        store = target.store
        vec = self.embedder.embed(content)

        new_id = f"{target.id}-{len(store.ids)}"
        all_ids = [*store.ids, new_id]
        all_contents = [*store.contents, content]
        all_vectors = vec.reshape(1, -1) if store.vectors is None else np.vstack([store.vectors, vec])

        target.store = InMemoryShardStore.from_vectors(all_ids, all_contents, all_vectors)
        target.centroid = _normalize(target.store.vectors.mean(axis=0))

    def search(
        self,
        query: str,
        budget: Budget | None = None,
        scope: str | None = None,
    ) -> AssembledContext:
        shards = self.shards_for_scope(scope)
        router = MemoryRouter(shards=shards, judge=self.judge)
        assembler = ContextAssembler(router=router, strategy=self.strategy, k=self.k)
        query_vec = _normalize(self.embedder.embed(query))
        return assembler.assemble(query, query_vec, budget or Budget(), scope=scope)
