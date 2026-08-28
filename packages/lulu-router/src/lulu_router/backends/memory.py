"""In-memory brute-force ShardStore.

Backing for tests and the DBpedia eval, where the corpus fits comfortably in
RAM and what's under test is routing behavior, not storage engineering.
The harness's production default (TurboVec + SQLite, ported from
local-memory/core/vector_store.py) lives in backends/turbovec_sqlite.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lulu_router.shard import SearchResult


@dataclass
class InMemoryShardStore:
    ids: list[str] = field(default_factory=list)
    contents: list[str] = field(default_factory=list)
    vectors: np.ndarray | None = None  # (n, dim), L2-normalized
    metadata: list[dict] = field(default_factory=list)

    def search(self, query_vec: np.ndarray, k: int, query: str = "") -> list[SearchResult]:
        # `query` (the original text) is unused here -- this backend only
        # ever compares vectors. It exists on the signature purely to
        # satisfy ShardStore for backends that DO need it (MCP-backed
        # shards, see shard.py's docstring).
        if self.vectors is None or len(self.ids) == 0:
            return []
        sims = self.vectors @ query_vec
        top = np.argsort(-sims)[:k]
        return [
            SearchResult(
                id=self.ids[i],
                content=self.contents[i],
                score=float(sims[i]),
                metadata=self.metadata[i] if i < len(self.metadata) else {},
            )
            for i in top
        ]

    def __len__(self) -> int:
        return len(self.ids)

    @classmethod
    def from_vectors(
        cls,
        ids: list[str],
        contents: list[str],
        vectors: np.ndarray,
        metadata: list[dict] | None = None,
    ) -> InMemoryShardStore:
        vecs = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        return cls(
            ids=list(ids),
            contents=list(contents),
            vectors=vecs,
            metadata=metadata or [{} for _ in ids],
        )
