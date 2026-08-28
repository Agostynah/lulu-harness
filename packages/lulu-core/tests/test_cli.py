"""cli.py: the wiring end-to-end, driven with FakeModelClient via
model_override so none of this needs a real Anthropic API key. The last
test is the actual point of this session's work: two "terminals" (two
separate `main()` invocations, same project root, different sessions)
racing to write the same file -- the second one must get escalated to a
human ASK instead of silently overwriting the first, even under a mode
that would normally auto-allow the write.
"""

from __future__ import annotations

from pathlib import Path

from lulu.cli import build_tool_registry, main, terminal_ask_human
from lulu.llm.client import ToolCall

from .fakes.model_client import FakeModelClient, text_response, tool_call_response


def test_build_tool_registry_registers_all_six_tools(tmp_path: Path):
    registry = build_tool_registry(tmp_path)
    names = {spec.name for spec in registry.specs()}
    assert names == {"read_file", "write_file", "edit_file", "glob", "grep", "bash"}


def test_terminal_ask_human_approves_on_y(monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda *_: "y")
    assert terminal_ask_human("write_file", {"path": "a.py"}, "some reason") is True
    assert "some reason" in capsys.readouterr().out


def test_terminal_ask_human_denies_on_anything_else(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *_: "")
    assert terminal_ask_human("write_file", {"path": "a.py"}, "reason") is False


def test_main_one_shot_prompt_runs_full_wiring(tmp_path: Path, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    model = FakeModelClient()
    model.queue(text_response("all done, no tools needed"))

    exit_code = main(["do the thing", "--root", str(root), "--mode", "manual"], model_override=model)

    assert exit_code == 0
    assert "all done" in capsys.readouterr().out
    sessions_dir = root / ".lulu" / "logs" / "sessions"
    assert len(list(sessions_dir.glob("*.jsonl"))) == 1


def test_main_writes_permissions_log_on_tool_use(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    model = FakeModelClient()
    model.queue(tool_call_response([ToolCall(id="tc1", name="write_file", arguments={"path": "a.py", "content": "x"})]))
    model.queue(text_response("done"))

    main(["write a file", "--root", str(root), "--mode", "auto_edits"], model_override=model)

    permissions_log = root / ".lulu" / "logs" / "permissions.jsonl"
    assert permissions_log.exists()
    assert (root / "a.py").read_text() == "x"


def test_main_two_racing_sessions_second_gets_asked_about_lock(tmp_path: Path, monkeypatch, capsys):
    """The actual adversarial scenario this feature exists for: two
    concurrent `lulu` processes (two terminals) pointed at the same
    project, both trying to write the same file. Session A writes first;
    session B -- even under auto_edits, which would normally ALLOW the
    write with no human involved at all -- must be escalated to ASK
    because session A's claim on the file is still fresh."""
    root = tmp_path / "proj"
    root.mkdir()

    model_a = FakeModelClient()
    model_a.queue(
        tool_call_response([ToolCall(id="tc1", name="write_file", arguments={"path": "shared.py", "content": "v1"})])
    )
    model_a.queue(text_response("done a"))
    main(["task A", "--root", str(root), "--mode", "auto_edits"], model_override=model_a)

    assert (root / "shared.py").read_text() == "v1"

    model_b = FakeModelClient()
    model_b.queue(
        tool_call_response([ToolCall(id="tc2", name="write_file", arguments={"path": "shared.py", "content": "v2"})])
    )
    model_b.queue(text_response("done b"))
    monkeypatch.setattr("builtins.input", lambda *_: "n")  # human denies -- v2 must never land

    main(["task B", "--root", str(root), "--mode", "auto_edits"], model_override=model_b)

    captured = capsys.readouterr()
    assert "locked by another session" in captured.out
    assert (root / "shared.py").read_text() == "v1"  # untouched by session B


# TTL expiry (a lock naturally stops mattering once it's stale) is covered
# directly and fast in test_locks.py / test_permissions_locks.py via a
# monkeypatched timestamp -- not re-tested here with a real 5-minute wait.
