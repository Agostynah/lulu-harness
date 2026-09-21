"""memory.py: MemoryStore -- the harness's actual local memory, wired
into AgentLoop (loop.py). Without this, ContextAssembler and MemoryRouter
are tested, working components that nothing in a real `lulu` run ever
calls -- this is the piece that makes the thesis true of the running
harness, not just of evals/dbpedia.

Backed by lulu_router's SQLiteShardStore when `data_dir` is given (the
harness's real default -- one .db file per shard under
<root>/.lulu/data/, survives a restart) or InMemoryShardStore when it
isn't (tests and anything else that shouldn't touch disk; `data_dir=None`
is the default for exactly that reason). The router and ContextAssembler
don't care which backs a Shard -- swapping the store only ever touches
this file and backends/.

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

import hashlib
import os
from pathlib import Path

import numpy as np
from lulu_router.backends.memory import InMemoryShardStore
from lulu_router.backends.sqlite import SQLiteShardStore
from lulu_router.cost import Budget, CostProfile
from lulu_router.judges.fallback import FallbackJudge
from lulu_router.judges.geometric import GeometricJudge
from lulu_router.judges.jev import JevJudge
from lulu_router.router import MemoryRouter
from lulu_router.shard import Shard
from lulu_router.strategies import Judge

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


def default_judge() -> Judge:
    """Jev whenever a key is configured, GeometricJudge otherwise -- Jev
    reads shard content and is cheap enough to be the default rather than
    an opt-in (see judges/jev.py), but it's an external API call, so it's
    wrapped in FallbackJudge: a real network failure or outage falls back
    to GeometricJudge (pure local math, can't be unreachable) for that
    round instead of taking the whole harness down. No JEV_API_KEY means
    no network call is ever attempted -- straight to GeometricJudge."""
    api_key = os.environ.get("JEV_API_KEY")
    if api_key:
        return FallbackJudge(primary=JevJudge(api_key=api_key), secondary=GeometricJudge())
    return GeometricJudge()


class MemoryStore:
    def __init__(
        self,
        embedder: Embedder | None = None,
        shard_costs: dict[str, CostProfile] | None = None,
        strategy: str = "progressive_expansion",
        k: int = 5,
        judge: Judge | None = None,
        data_dir: Path | None = None,
    ) -> None:
        self.embedder = embedder or Embedder()
        self.shard_costs = shard_costs or DEFAULT_SHARD_COSTS
        self.strategy = strategy
        self.k = k
        self.judge = judge or default_judge()
        self.data_dir = data_dir
        # Keyed on (shard_type, scope) -- NOT just shard_type. This is
        # the actual fix: two different scopes writing to "episodic"
        # get two distinct Shard objects (distinct store instances too),
        # never one shared store with a unioned permission set. See
        # module docstring.
        self._shards: dict[tuple[str, str | None], Shard] = {}

    def _shard(self, shard_type: str, scope: str | None) -> Shard:
        if shard_type not in self.shard_costs:
            raise ValueError(f"unknown shard {shard_type!r}; known shards: {list(self.shard_costs)}")
        key = (shard_type, scope)
        if key not in self._shards:
            shard_id = shard_type if scope is None else f"{shard_type}:{scope}"
            store = self._build_store(shard_type, scope)
            # A SQLiteShardStore opened against a file from a previous run
            # already has data before this process ever calls write() --
            # without seeding the centroid here, a shard with real
            # persisted memories would look centroid-less (and therefore
            # get skipped by every routing strategy that checks it) until
            # something happens to write to it again this session.
            centroid_fn = getattr(store, "centroid", None)
            raw_centroid = centroid_fn() if callable(centroid_fn) else None
            self._shards[key] = Shard(
                id=shard_id,
                store=store,
                cost=self.shard_costs[shard_type],
                centroid=_normalize(raw_centroid) if raw_centroid is not None else None,
                allowed_scopes=None if scope is None else frozenset({scope}),
            )
        return self._shards[key]

    def _shard_db_path(self, shard_type: str, scope: str | None) -> Path:
        # scope is caller-supplied and ends up in a filename -- hashed
        # rather than interpolated raw, so an adversarial scope string
        # can't do anything path-traversal-shaped to it (the same
        # discipline CONTRIBUTING.md asks for path handling in general;
        # session.py had a real bug in this family before session_id
        # validation was added). shard_type is never attacker-controlled
        # -- it's one of self.shard_costs' fixed keys, already validated
        # by the only caller (_shard()) -- so it's safe to use as-is.
        assert self.data_dir is not None
        if scope is None:
            filename = f"{shard_type}.db"
        else:
            scope_hash = hashlib.sha256(scope.encode()).hexdigest()[:16]
            filename = f"{shard_type}__{scope_hash}.db"
        return self.data_dir / filename

    def _build_store(self, shard_type: str, scope: str | None):
        if self.data_dir is None:
            return InMemoryShardStore()
        return SQLiteShardStore(self._shard_db_path(shard_type, scope))

    def shards_for_scope(self, scope: str | None) -> list[Shard]:
        """Every shard a caller with this scope may legitimately search --
        its own scope's shards, and only its own, regardless of how many
        other scopes happen to also have written to a shard of the same
        *type*. Used both by search() and by callers rendering /cost
        (cli.py, server.py), which must show a scope-appropriate
        counterfactual rather than one that includes other scopes'
        shards -- that would itself leak "how much data exists for other
        tenants," a subtler version of the same class of bug.

        Touches (loads) every known shard_type for this scope that already
        has a .db file on disk, first (_shard() is idempotent -- a no-op
        if already loaded). Without this, a shard whose SQLiteShardStore
        already has data from a previous run stays invisible to search()
        until something calls write() to it THIS session -- self._shards
        only ever grows lazily, it never discovers what already exists on
        disk on its own. Only pre-touches shard types with an existing
        file, not every known type unconditionally -- in-memory backends
        (data_dir=None, tests) have nothing to discover, and a shard type
        genuinely never written to shouldn't materialize as an empty,
        pointless-to-contact Shard just because search() ran."""
        if self.data_dir is not None:
            for shard_type in self.shard_costs:
                if self._shard_db_path(shard_type, scope).exists():
                    self._shard(shard_type, scope)
        return [shard for (_shard_type, shard_scope), shard in self._shards.items() if shard_scope == scope]

    def write(self, content: str, shard: str = "episodic", scope: str | None = None) -> None:
        target = self._shard(shard, scope)
        vec = self.embedder.embed(content)
        new_id = f"{target.id}-{len(target.store)}"
        # add() is O(log n) for SQLiteShardStore (a real INSERT) and O(n)
        # for InMemoryShardStore (numpy arrays aren't append-friendly) --
        # either way, no longer the O(n) full-shard rebuild per write this
        # used to be (see backends/memory.py's InMemoryShardStore.add()
        # docstring for what that cost).
        target.store.add(new_id, content, vec)
        target.centroid = _normalize(target.store.centroid())

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
