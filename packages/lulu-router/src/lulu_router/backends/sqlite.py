"""SQLiteShardStore: the harness's persistent local ShardStore.

InMemoryShardStore (backends/memory.py) is honestly in-process only --
memory.py's own module docstring calls this out as the harness's biggest
v0 shortcut: restart the process and every memory is gone. This backend
fixes that with the simplest thing that could work at the harness's
actual scale (memory.py's own words: "a harness's day-to-day memory
starts small... clustering it would be solving a problem that doesn't
exist yet") -- one SQLite file per shard, vectors stored as raw float32
blobs, brute-force cosine similarity over whatever's in the table. No ANN
index, no KMeans, no ONNX/native dependency beyond the stdlib `sqlite3`
module already in every Python install.

Also fixes memory.py's write() being O(n) per call (InMemoryShardStore is
immutable, so every write there rebuilds the whole shard from scratch --
see that file's comment on it). `add()` here is a single SQLite INSERT,
O(log n) for the index, not O(n) for the whole store.

Reads cache the loaded (ids, contents, vectors) matrix in-process and
invalidate it on the next `add()` -- avoids re-querying and
re-deserializing every row on every single search call within a session,
without reaching for anything heavier than "don't redo work that didn't
change."
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np

from lulu_router.shard import SearchResult


class SQLiteShardStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        # WAL: readers (search) don't block a concurrent writer (add), and
        # vice versa -- the default rollback-journal mode takes an
        # exclusive lock for the whole write transaction, which would
        # stall a search running at the same time in another session.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vectors (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                vector BLOB NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self._conn.commit()
        self._cache: tuple[list[str], list[str], np.ndarray, list[dict]] | None = None

    def add(self, id: str, content: str, vector: np.ndarray, metadata: dict | None = None) -> None:
        # Normalized to unit length before storing -- search()'s
        # `vectors @ query_vec` is only a real cosine similarity if both
        # sides are unit vectors (memory.py normalizes the query side).
        vec = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec = vec / norm
        self._conn.execute(
            "INSERT INTO vectors (id, content, vector, metadata) VALUES (?, ?, ?, ?)",
            (id, content, vec.tobytes(), json.dumps(metadata or {})),
        )
        self._conn.commit()
        self._cache = None  # stale -- rebuilt lazily on the next search()

    def _load(self) -> tuple[list[str], list[str], np.ndarray, list[dict]]:
        if self._cache is not None:
            return self._cache
        rows = self._conn.execute("SELECT id, content, vector, metadata FROM vectors").fetchall()
        ids = [r[0] for r in rows]
        contents = [r[1] for r in rows]
        vectors = (
            np.stack([np.frombuffer(r[2], dtype=np.float32) for r in rows])
            if rows
            else np.zeros((0, 0), dtype=np.float32)
        )
        metadata = [json.loads(r[3]) for r in rows]
        self._cache = (ids, contents, vectors, metadata)
        return self._cache

    def search(self, query_vec: np.ndarray, k: int, query: str = "") -> list[SearchResult]:
        # `query` unused here for the same reason as InMemoryShardStore's
        # -- see shard.py's ShardStore docstring.
        ids, contents, vectors, metadata = self._load()
        if len(ids) == 0:
            return []
        sims = vectors @ np.asarray(query_vec, dtype=np.float32)
        top = np.argsort(-sims)[:k]
        return [
            SearchResult(id=ids[i], content=contents[i], score=float(sims[i]), metadata=metadata[i]) for i in top
        ]

    def centroid(self) -> np.ndarray | None:
        """Mean of every stored vector, or None if the shard is empty --
        used to seed Shard.centroid on load so a shard that already has
        data from a previous run doesn't look centroid-less (and
        therefore unrouted-to) until the next write happens to touch it."""
        _ids, _contents, vectors, _metadata = self._load()
        if vectors.shape[0] == 0:
            return None
        return vectors.mean(axis=0)

    def __len__(self) -> int:
        (count,) = self._conn.execute("SELECT COUNT(*) FROM vectors").fetchone()
        return count
