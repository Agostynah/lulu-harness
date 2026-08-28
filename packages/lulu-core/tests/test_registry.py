"""ToolRegistry: registration, spec export, and dispatch -- including the
guarantee that a handler raising never crashes dispatch, it turns into an
error ToolResult instead."""

from __future__ import annotations

from lulu.llm.client import ToolCall
from lulu.tools.base import Tool
from lulu.tools.registry import ToolRegistry


def _tool(name: str, handler) -> Tool:
    return Tool(name=name, description=f"{name} tool", input_schema={"type": "object"}, handler=handler)


def test_register_and_specs_round_trip():
    registry = ToolRegistry()
    registry.register(_tool("echo", lambda args: args["text"]))

    specs = registry.specs()

    assert len(specs) == 1
    assert specs[0].name == "echo"
    assert specs[0].description == "echo tool"


def test_dispatch_returns_handler_output():
    registry = ToolRegistry()
    registry.register(_tool("echo", lambda args: args["text"]))

    result = registry.dispatch(ToolCall(id="tc_1", name="echo", arguments={"text": "hi"}))

    assert result.tool_call_id == "tc_1"
    assert result.content == "hi"
    assert result.is_error is False


def test_dispatch_unknown_tool_returns_error_result_not_exception():
    registry = ToolRegistry()

    result = registry.dispatch(ToolCall(id="tc_1", name="does_not_exist", arguments={}))

    assert result.is_error is True
    assert "unknown tool" in result.content


def test_dispatch_handler_exception_becomes_error_result_not_a_crash():
    def boom(args):
        raise ValueError("something bad")

    registry = ToolRegistry()
    registry.register(_tool("boom", boom))

    result = registry.dispatch(ToolCall(id="tc_1", name="boom", arguments={}))

    assert result.is_error is True
    assert "ValueError" in result.content
    assert "something bad" in result.content


def test_dispatch_preserves_tool_call_id_on_error():
    registry = ToolRegistry()
    result = registry.dispatch(ToolCall(id="tc_specific_id", name="missing", arguments={}))
    assert result.tool_call_id == "tc_specific_id"
