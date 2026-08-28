"""connectors/mcp.py: wraps a real MCP server connection as a ShardStore
with a MEASURED CostProfile, not a simulated one.

This is what takes the `remote` shard in evals/dbpedia and
evals/agent_tasks from hypothetical to real. See docs/THESIS.md: the
shard's entire reason for existing is that its cost isn't uniform with
the local shards, and that claim is only honest if the cost is actually
measured, not guessed.

This module never starts a transport itself -- it's handed an
already-connected `mcp.ClientSession` (stdio for a locally-spawned server,
SSE/HTTP for a remote one), so it never has to know how the server was
launched. That's also why it's typed loosely against `session: Any`
rather than importing `mcp.ClientSession` for the type hint -- nothing
here needs to know more about the SDK than "has an async call_tool()".

Known limitation, stated rather than papered over: `search()` (the sync
method lulu_router's ShardStore Protocol requires) bridges to the async
SDK via `asyncio.run()`, which starts a fresh event loop per call. That's
fine for today's harness, which makes one sync call at a time with no
event loop already running -- it will NOT work if this is ever called
from inside an already-running loop (e.g. once the harness's core loop
goes async, likely alongside real SSE streaming). `search_async()` is the
real implementation; `search()` is a v0-only bridge on top of it.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any

from lulu_router.shard import SearchResult


class McpToolError(Exception):
    """The connected MCP server's tool call itself reported an error
    (result.is_error) -- distinct from a transport/protocol failure,
    which the mcp SDK would raise as its own exception type and which
    this module doesn't catch or reinterpret."""


@dataclass
class McpShardStore:
    session: Any
    tool_name: str
    query_arg: str = "query"
    result_limit_arg: str | None = "limit"
    _latencies_ms: list[float] = field(default_factory=list, repr=False)

    async def search_async(self, k: int, query: str) -> list[SearchResult]:
        arguments: dict[str, Any] = {self.query_arg: query}
        if self.result_limit_arg:
            arguments[self.result_limit_arg] = k

        t0 = time.perf_counter()
        result = await self.session.call_tool(self.tool_name, arguments)
        self._latencies_ms.append((time.perf_counter() - t0) * 1000)

        if getattr(result, "is_error", False):
            texts = [c.text for c in result.content if getattr(c, "type", None) == "text"]
            raise McpToolError(f"MCP tool {self.tool_name!r} returned an error: {'; '.join(texts)}")

        results: list[SearchResult] = []
        for i, block in enumerate(result.content):
            if getattr(block, "type", None) != "text":
                continue
            results.append(SearchResult(id=f"{self.tool_name}-{i}", content=block.text, score=1.0 / (i + 1)))
        return results[:k]

    def search(self, query_vec: Any, k: int, query: str = "") -> list[SearchResult]:
        """Sync bridge for ShardStore compliance -- see module docstring
        for the event-loop caveat. `query_vec` is accepted but unused: an
        MCP-backed shard has no local embedding space to compare a vector
        against, it searches by the original text instead (see
        lulu_router.shard.ShardStore's docstring for why `query` exists
        on the Protocol at all)."""
        return asyncio.run(self.search_async(k, query))

    def __len__(self) -> int:
        # A remote shard's size isn't locally knowable without a separate
        # count call most servers don't expose. 0 signals "unknown size",
        # not "empty" -- routing strategies never rely on len() for
        # anything besides InMemoryShardStore's own bookkeeping.
        return 0

    def measured_latency_ms(self) -> float | None:
        """Median latency across calls made so far, or None before the
        first call. Feed this into a real CostProfile once there's a
        representative sample -- guessing this number would defeat the
        entire point of a MEASURED remote shard (docs/THESIS.md)."""
        if not self._latencies_ms:
            return None
        ordered = sorted(self._latencies_ms)
        return ordered[len(ordered) // 2]
