"""A minimal double for the one MCP SDK surface McpShardStore actually
touches: ClientSession.call_tool(). No real transport, no real server --
just enough shape to drive search_async()'s logic deterministically."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class FakeTextContent:
    text: str
    type: str = "text"


@dataclass
class FakeCallToolResult:
    content: list[Any]
    is_error: bool = False


@dataclass
class FakeClientSession:
    """Queue-based: each call_tool() pops the next queued result (or
    raises if it's an Exception instance), and records every call's
    (tool_name, arguments) for assertions."""

    _queue: list[Any] = field(default_factory=list)
    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def queue(self, result: Any) -> None:
        self._queue.append(result)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeCallToolResult:
        self.calls.append((name, arguments))
        result = self._queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result
