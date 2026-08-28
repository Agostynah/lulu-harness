"""Shard: the unit the router reasons about.

A shard bundles a store (where vectors/payloads actually live) with a
CostProfile (what it costs to ask it a question) and an access policy (who
is allowed to ask). Storage is deliberately hidden behind the ShardStore
protocol so the router never has to know whether a shard is a local SQLite
table, an in-memory dict, or an MCP connector to a remote service -- it
only ever sees `centroid`, `cost`, and `search()`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from lulu_router.cost import CostProfile


@dataclass
class SearchResult:
    id: str
    content: str
    score: float  # cosine similarity to the query, in [-1, 1]
    metadata: dict = field(default_factory=dict)


class ShardStore(Protocol):
    """What a shard's storage backend must provide.

    Implementations: `backends.memory.InMemoryShardStore` (brute-force,
    used for tests and the DBpedia eval), a TurboVec+SQLite backend (the
    harness's local default, ported from local-memory/core/vector_store.py),
    and an MCP connector backend (the harness's `remote` shard).

    `search` takes both the query vector AND the original query text.
    Local vector backends only need the vector; a remote MCP-backed shard
    (lulu.connectors.mcp) has no local embedding space to compare against
    and calls a remote tool with the text instead -- `query` exists
    specifically so that backend isn't stuck reverse-engineering text from
    a vector it can't decode. Vector-only backends simply ignore it.
    """

    def search(self, query_vec: np.ndarray, k: int, query: str = "") -> list[SearchResult]: ...

    def __len__(self) -> int: ...


@dataclass
class Shard:
    """A named, costed, searchable partition of memory."""

    id: str
    store: ShardStore
    cost: CostProfile
    centroid: np.ndarray | None = None  # mean embedding of the shard's contents
    allowed_scopes: frozenset[str] | None = None  # None = unrestricted

    def permits(self, scope: str | None) -> bool:
        """Access-policy check. A shard with allowed_scopes=None is open to
        any caller; otherwise the caller's scope must be explicitly listed.

        This is the mechanism a flat index cannot express: once vectors are
        merged into one index there is nowhere left to attach a boundary
        like "never route customer A's queries into customer B's memory."
        """
        if self.allowed_scopes is None:
            return True
        return scope in self.allowed_scopes

    def search(self, query_vec: np.ndarray, k: int, query: str = "") -> list[SearchResult]:
        return self.store.search(query_vec, k, query=query)

    def __len__(self) -> int:
        return len(self.store)
