"""AgentLoop: the reason -> act -> observe -> repeat cycle, permission
gating (DENY never asks, ASK calls ask_human and respects the answer,
ALLOW never asks), and the adversarial case that matters most for a
harness -- a model that never stops calling tools must not hang the
loop forever."""

from __future__ import annotations

from pathlib import Path


from lulu.attention import AttentionMode
from lulu.llm.client import Message, ToolCall
from lulu.loop import AgentLoop
from lulu.permissions import PermissionChecker
from lulu.tools.base import Tool
from lulu.tools.registry import ToolRegistry
from .fakes.model_client import FakeModelClient, text_response, tool_call_response


def _registry(handler=None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="write_file",
            description="writes a file",
            input_schema={"type": "object"},
            handler=handler or (lambda args: f"wrote {args.get('path')}"),
        )
    )
    return registry


def test_final_text_response_stops_immediately():
    model = FakeModelClient()
    model.queue(text_response("all done"))
    loop = AgentLoop(
        model=model,
        tools=_registry(),
        permissions=PermissionChecker(mode=AttentionMode.MANUAL),
        ask_human=lambda *a: True,
    )

    result = loop.run_turn([], "do something")

    assert result.stopped_reason == "final_text"
    assert result.iterations == 1
    assert result.messages[-1].content == "all done"


def test_allow_mode_executes_tool_without_asking_human():
    model = FakeModelClient()
    model.queue(tool_call_response([ToolCall(id="tc_1", name="write_file", arguments={"path": "a.py"})]))
    model.queue(text_response("done"))
    ask_calls = []

    loop = AgentLoop(
        model=model,
        tools=_registry(),
        permissions=PermissionChecker(mode=AttentionMode.AUTO),  # write_file = ALLOW
        ask_human=lambda name, args, reason: ask_calls.append((name, args)) or True,
    )

    result = loop.run_turn([], "write a file")

    assert ask_calls == []  # never asked -- ALLOW doesn't consult the human
    assert result.iterations == 2
    tool_result_message = result.messages[-2]
    assert tool_result_message.tool_results[0].content == "wrote a.py"
    assert tool_result_message.tool_results[0].is_error is False


def test_ask_decision_calls_ask_human_and_executes_on_approval():
    model = FakeModelClient()
    model.queue(tool_call_response([ToolCall(id="tc_1", name="write_file", arguments={"path": "a.py"})]))
    model.queue(text_response("done"))

    loop = AgentLoop(
        model=model,
        tools=_registry(),
        permissions=PermissionChecker(mode=AttentionMode.MANUAL),  # write_file = ASK
        ask_human=lambda name, args, reason: True,
    )

    result = loop.run_turn([], "write a file")

    tool_result_message = result.messages[-2]
    assert tool_result_message.tool_results[0].is_error is False
    assert tool_result_message.tool_results[0].content == "wrote a.py"


def test_ask_decision_denied_by_human_does_not_execute_tool():
    handler_calls = []

    def handler(args):
        handler_calls.append(args)
        return "should not run"

    model = FakeModelClient()
    model.queue(tool_call_response([ToolCall(id="tc_1", name="write_file", arguments={"path": "a.py"})]))
    model.queue(text_response("ok, skipped"))

    loop = AgentLoop(
        model=model,
        tools=_registry(handler),
        permissions=PermissionChecker(mode=AttentionMode.MANUAL),
        ask_human=lambda name, args, reason: False,
    )

    result = loop.run_turn([], "write a file")

    assert handler_calls == []  # the tool handler itself was never invoked
    tool_result_message = result.messages[-2]
    assert tool_result_message.tool_results[0].is_error is True
    assert "denied by user" in tool_result_message.tool_results[0].content


def test_deny_mode_refuses_without_ever_asking_human():
    ask_calls = []
    model = FakeModelClient()
    model.queue(tool_call_response([ToolCall(id="tc_1", name="write_file", arguments={"path": "a.py"})]))
    model.queue(text_response("blocked"))

    loop = AgentLoop(
        model=model,
        tools=_registry(),
        permissions=PermissionChecker(mode=AttentionMode.PLAN),  # write_file = DENY
        ask_human=lambda name, args, reason: ask_calls.append(1) or True,
    )

    result = loop.run_turn([], "write a file")

    assert ask_calls == []  # DENY is decided by policy alone, never escalated to a human
    tool_result_message = result.messages[-2]
    assert tool_result_message.tool_results[0].is_error is True
    assert "denied by policy" in tool_result_message.tool_results[0].content


def test_model_that_never_stops_calling_tools_is_capped_at_max_iterations():
    """Adversarial: a model (buggy, or adversarially prompted) that keeps
    emitting tool calls forever must not hang the harness. The loop has
    to give up after max_iterations, not loop until the process is killed."""
    model = FakeModelClient()
    for _ in range(10):
        model.queue(tool_call_response([ToolCall(id="tc_x", name="write_file", arguments={"path": "x"})]))

    loop = AgentLoop(
        model=model,
        tools=_registry(),
        permissions=PermissionChecker(mode=AttentionMode.AUTO),
        ask_human=lambda *a: True,
        max_iterations=3,
    )

    result = loop.run_turn([], "loop forever")

    assert result.stopped_reason == "max_iterations"
    assert result.iterations == 3
    assert len(model.calls) == 3  # never called a 4th time


def test_multiple_tool_calls_in_one_response_are_all_executed():
    model = FakeModelClient()
    model.queue(
        tool_call_response(
            [
                ToolCall(id="tc_1", name="write_file", arguments={"path": "a.py"}),
                ToolCall(id="tc_2", name="write_file", arguments={"path": "b.py"}),
            ]
        )
    )
    model.queue(text_response("both done"))

    loop = AgentLoop(
        model=model,
        tools=_registry(),
        permissions=PermissionChecker(mode=AttentionMode.AUTO),
        ask_human=lambda *a: True,
    )

    result = loop.run_turn([], "write two files")

    tool_result_message = result.messages[-2]
    assert len(tool_result_message.tool_results) == 2
    assert {r.tool_call_id for r in tool_result_message.tool_results} == {"tc_1", "tc_2"}


def test_permission_decisions_are_logged(tmp_path: Path):
    log_path = tmp_path / "permissions.jsonl"
    model = FakeModelClient()
    model.queue(tool_call_response([ToolCall(id="tc_1", name="write_file", arguments={"path": "a.py"})]))
    model.queue(text_response("done"))

    loop = AgentLoop(
        model=model,
        tools=_registry(),
        permissions=PermissionChecker(mode=AttentionMode.MANUAL, log_path=log_path),
        ask_human=lambda *a: True,
    )

    loop.run_turn([], "write a file")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"outcome": "approved"' in lines[0]


def test_history_is_preserved_and_extended():
    model = FakeModelClient()
    model.queue(text_response("second reply"))

    prior_history = [Message(role="user", content="first"), Message(role="assistant", content="first reply")]
    loop = AgentLoop(
        model=model,
        tools=_registry(),
        permissions=PermissionChecker(mode=AttentionMode.MANUAL),
        ask_human=lambda *a: True,
    )

    result = loop.run_turn(prior_history, "second question")

    assert result.messages[0].content == "first"
    assert result.messages[1].content == "first reply"
    assert result.messages[2].content == "second question"
    assert result.messages[3].content == "second reply"
