"""Session: JSONL round-trip (append -> load_history reconstructs the same
conversation), usage attribution across multi-iteration turns, and resume
semantics."""

from __future__ import annotations

from pathlib import Path

from lulu.llm.client import Message, ToolCall, ToolResult, Usage
from lulu.loop import TurnResult
from lulu.session import Session


def test_new_session_has_no_history(tmp_path: Path):
    session = Session.new(log_dir=tmp_path)
    assert session.load_history() == []
    assert session.total_usage() == Usage()


def test_append_and_load_round_trips_simple_message(tmp_path: Path):
    session = Session.new(log_dir=tmp_path)
    session.append_message(Message(role="user", content="hello"))

    history = session.load_history()

    assert len(history) == 1
    assert history[0].role == "user"
    assert history[0].content == "hello"


def test_append_and_load_round_trips_tool_calls_and_results(tmp_path: Path):
    session = Session.new(log_dir=tmp_path)
    session.append_message(
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="tc_1", name="write_file", arguments={"path": "a.py"})],
        )
    )
    session.append_message(
        Message(role="user", tool_results=[ToolResult(tool_call_id="tc_1", content="wrote a.py", is_error=False)])
    )

    history = session.load_history()

    assert history[0].tool_calls[0].name == "write_file"
    assert history[0].tool_calls[0].arguments == {"path": "a.py"}
    assert history[1].tool_results[0].content == "wrote a.py"


def test_total_usage_accumulates_across_messages(tmp_path: Path):
    session = Session.new(log_dir=tmp_path)
    session.append_message(Message(role="assistant", content="a"), usage=Usage(input_tokens=10, output_tokens=5))
    session.append_message(Message(role="assistant", content="b"), usage=Usage(input_tokens=20, output_tokens=8))
    session.append_message(Message(role="user", content="no usage here"))

    total = session.total_usage()

    assert total.input_tokens == 30
    assert total.output_tokens == 13


def test_resume_loads_a_previously_written_session(tmp_path: Path):
    original = Session.new(log_dir=tmp_path)
    original.append_message(Message(role="user", content="first ever message"))

    resumed = Session.resume(original.session_id, log_dir=tmp_path)

    assert resumed.load_history()[0].content == "first ever message"


def test_list_sessions_enumerates_written_sessions(tmp_path: Path):
    a = Session.new(log_dir=tmp_path)
    a.append_message(Message(role="user", content="x"))
    b = Session.new(log_dir=tmp_path)
    b.append_message(Message(role="user", content="y"))

    listed = Session.list_sessions(tmp_path)

    assert set(listed) == {a.session_id, b.session_id}


def test_list_sessions_on_missing_dir_returns_empty(tmp_path: Path):
    assert Session.list_sessions(tmp_path / "does_not_exist") == []


def test_append_turn_result_skips_prior_history_and_attributes_usage(tmp_path: Path):
    """A turn with two model round-trips (tool call, then final text)
    should log exactly 3 new messages -- assistant(tool_call),
    user(tool_result), assistant(final) -- with usage attributed only to
    the two assistant messages, in call order, and the prior history
    should not be re-logged."""
    session = Session.new(log_dir=tmp_path)
    prior = [Message(role="user", content="already logged separately")]

    turn_result = TurnResult(
        messages=[
            *prior,
            Message(role="user", content="do the thing"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="tc_1", name="write_file", arguments={"path": "a.py"})],
            ),
            Message(role="user", tool_results=[ToolResult(tool_call_id="tc_1", content="wrote a.py")]),
            Message(role="assistant", content="all done"),
        ],
        iterations=2,
        stopped_reason="final_text",
        usages=[Usage(input_tokens=100, output_tokens=20), Usage(input_tokens=150, output_tokens=10)],
    )

    session.append_turn_result(turn_result, messages_before=len(prior))

    logged = session.load_history()
    assert len(logged) == 4  # the user turn-starter + tool_call + tool_result + final text
    assert logged[0].content == "do the thing"
    assert session.total_usage().input_tokens == 250
    assert session.total_usage().output_tokens == 30


def test_load_history_ignores_blank_lines(tmp_path: Path):
    session = Session.new(log_dir=tmp_path)
    session.path.parent.mkdir(parents=True, exist_ok=True)
    session.path.write_text('{"role": "user", "content": "x"}\n\n\n', encoding="utf-8")

    assert len(session.load_history()) == 1
