"""In-memory brute-force ShardStore.

Backing for tests and the DBpedia eval, where the corpus fits comfortably in
RAM and what's under test is routing behavior, not storage engineering.
The harness's production default -- persistent, survives a restart --
is `backends.sqlite.SQLiteShardStore`.
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

    def add(self, id: str, content: str, vector: np.ndarray, metadata: dict | None = None) -> None:
        """O(n) -- rebuilds the whole vector array, since numpy arrays
        aren't append-friendly. Fine for this backend's actual job (tests,
        the DBpedia eval's fixed corpus); SQLiteShardStore.add() is the
        real O(log n) path production code should use instead. Kept as a
        method (not memory.py reaching into .ids/.contents/.vectors
        directly, as it used to) so MemoryStore.write() can treat either
        backend the same way.

        Normalizes `vector` to unit length before storing -- search()'s
        `self.vectors @ query_vec` is only a real cosine similarity if
        both sides are unit vectors; memory.py normalizes the query side,
        this is the store side of that same contract (matches
        from_vectors()'s per-row normalization, just one row at a time)."""
        vec = np.asarray(vector, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        self.ids.append(id)
        self.contents.append(content)
        self.metadata.append(metadata or {})
        self.vectors = vec if self.vectors is None else np.vstack([self.vectors, vec])

    def centroid(self) -> np.ndarray | None:
        return self.vectors.mean(axis=0) if self.vectors is not None and len(self.ids) > 0 else None

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
