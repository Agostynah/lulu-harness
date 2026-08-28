"""ToolRegistry: name -> Tool lookup, JSON-schema export for the
ModelClient, and dispatch. Dispatch here NEVER decides whether a call is
*allowed* -- loop.py calls permissions.py's check before it ever calls
dispatch(). This is purely the name -> execution -> ToolResult mapping.
"""

from __future__ import annotations

from lulu.llm.client import ToolCall, ToolResult, ToolSpec
from lulu.tools.base import Tool


class UnknownToolError(Exception):
    pass


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=t.name, description=t.description, input_schema=t.input_schema)
            for t in self._tools.values()
        ]

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise UnknownToolError(name) from None

    def dispatch(self, call: ToolCall) -> ToolResult:
        try:
            tool = self.get(call.name)
        except UnknownToolError:
            return ToolResult(tool_call_id=call.id, content=f"unknown tool: {call.name!r}", is_error=True)

        try:
            output = tool.handler(call.arguments)
        except Exception as exc:
            # A tool handler raising is an expected, common outcome (file
            # not found, path escapes root, ambiguous edit target) -- it
            # must never crash the agent loop. Convert to a visible
            # ToolResult error so the model can see what went wrong and
            # react, rather than the whole turn dying.
            return ToolResult(tool_call_id=call.id, content=f"{type(exc).__name__}: {exc}", is_error=True)

        return ToolResult(tool_call_id=call.id, content=output, is_error=False)
