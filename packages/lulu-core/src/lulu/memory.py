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
        costs = shard_costs or DEFAULT_SHARD_COSTS
        self._shards: dict[str, Shard] = {
            name: Shard(id=name, store=InMemoryShardStore(), cost=cost) for name, cost in costs.items()
        }
        self.router = MemoryRouter(shards=list(self._shards.values()), judge=GeometricJudge())
        self.assembler = ContextAssembler(router=self.router, strategy=strategy, k=k)

    def write(self, content: str, shard: str = "episodic", scope: str | None = None) -> None:
        if shard not in self._shards:
            raise ValueError(f"unknown shard {shard!r}; known shards: {list(self._shards)}")
        target = self._shards[shard]
        store = target.store
        vec = self.embedder.embed(content)

        new_id = f"{shard}-{len(store.ids)}"
        all_ids = [*store.ids, new_id]
        all_contents = [*store.contents, content]
        all_vectors = vec.reshape(1, -1) if store.vectors is None else np.vstack([store.vectors, vec])

        target.store = InMemoryShardStore.from_vectors(all_ids, all_contents, all_vectors)
        target.centroid = _normalize(target.store.vectors.mean(axis=0))
        if scope is not None:
            existing = target.allowed_scopes or frozenset()
            target.allowed_scopes = existing | {scope}

    def search(
        self,
        query: str,
        budget: Budget | None = None,
        scope: str | None = None,
    ) -> AssembledContext:
        query_vec = _normalize(self.embedder.embed(query))
        return self.assembler.assemble(query, query_vec, budget or Budget(), scope=scope)
