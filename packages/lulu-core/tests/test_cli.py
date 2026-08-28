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

import pytest

from lulu.cli import build_model_client, build_tool_registry, main, terminal_ask_human
from lulu.config import LuluConfig
from lulu.llm.anthropic_client import AnthropicClient
from lulu.llm.client import ToolCall
from lulu.llm.ollama_client import OllamaClient
from lulu.llm.openrouter_client import OpenRouterClient
from lulu.memory import MemoryStore

from .fakes.embedder import ZeroEmbedder
from .fakes.model_client import FakeModelClient, text_response, tool_call_response


def _fake_memory() -> MemoryStore:
    """A real MemoryStore, but wired to ZeroEmbedder so AgentLoop's
    memory.search()/.write() calls never try to load the actual
    embedding model -- these tests care about CLI wiring, not retrieval
    quality (that's test_memory.py's job)."""
    return MemoryStore(embedder=ZeroEmbedder())


def test_build_tool_registry_registers_all_six_tools(tmp_path: Path):
    registry = build_tool_registry(tmp_path)
    names = {spec.name for spec in registry.specs()}
    assert names == {"read_file", "write_file", "edit_file", "glob", "grep", "bash"}


def test_build_model_client_defaults_to_anthropic():
    client = build_model_client(LuluConfig(provider="anthropic"))
    assert isinstance(client, AnthropicClient)


def test_build_model_client_dispatches_to_openrouter(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = build_model_client(LuluConfig(provider="openrouter", model="anthropic/claude-sonnet-5"))
    assert isinstance(client, OpenRouterClient)
    assert client.model == "anthropic/claude-sonnet-5"


def test_build_model_client_dispatches_to_ollama():
    client = build_model_client(LuluConfig(provider="ollama", model="llama3.1"))
    assert isinstance(client, OllamaClient)
    assert client.model == "llama3.1"


def test_provider_flag_overrides_config_default(tmp_path: Path, monkeypatch, capsys):
    """--provider openrouter should switch the dispatch even with no
    lulu.toml present (whose default would be anthropic) -- verified by
    checking main() fails on the *right* reason (no OPENROUTER_API_KEY),
    proving OpenRouterClient's construction path was actually reached
    rather than falling through to Anthropic."""
    root = tmp_path / "proj"
    root.mkdir()
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        main(["hi", "--root", str(root), "--provider", "openrouter"])


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

    exit_code = main(
        ["do the thing", "--root", str(root), "--mode", "manual"],
        model_override=model,
        memory_override=_fake_memory(),
    )

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

    main(
        ["write a file", "--root", str(root), "--mode", "auto_edits"],
        model_override=model,
        memory_override=_fake_memory(),
    )

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
    main(
        ["task A", "--root", str(root), "--mode", "auto_edits"],
        model_override=model_a,
        memory_override=_fake_memory(),
    )

    assert (root / "shared.py").read_text() == "v1"

    model_b = FakeModelClient()
    model_b.queue(
        tool_call_response([ToolCall(id="tc2", name="write_file", arguments={"path": "shared.py", "content": "v2"})])
    )
    model_b.queue(text_response("done b"))
    monkeypatch.setattr("builtins.input", lambda *_: "n")  # human denies -- v2 must never land

    main(
        ["task B", "--root", str(root), "--mode", "auto_edits"],
        model_override=model_b,
        memory_override=_fake_memory(),
    )

    captured = capsys.readouterr()
    assert "locked by another session" in captured.out
    assert (root / "shared.py").read_text() == "v1"  # untouched by session B


# TTL expiry (a lock naturally stops mattering once it's stale) is covered
# directly and fast in test_locks.py / test_permissions_locks.py via a
# monkeypatched timestamp -- not re-tested here with a real 5-minute wait.


def _scripted_input(monkeypatch: pytest.MonkeyPatch, lines: list[str]) -> None:
    remaining = iter(lines)
    monkeypatch.setattr("builtins.input", lambda *_: next(remaining))


def test_trace_command_before_any_turn_shows_placeholder(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    _scripted_input(monkeypatch, ["/trace", "exit"])

    main(["--root", str(root)], model_override=FakeModelClient(), memory_override=_fake_memory())

    assert "no trace yet" in capsys.readouterr().out


def test_trace_and_cost_commands_render_after_a_real_turn(tmp_path: Path, monkeypatch, capsys):
    root = tmp_path / "proj"
    root.mkdir()
    model = FakeModelClient()
    model.queue(text_response("hi there"))
    _scripted_input(monkeypatch, ["hello", "/trace", "/cost", "exit"])

    main(["--root", str(root)], model_override=model, memory_override=_fake_memory())

    output = capsys.readouterr().out
    assert "/trace" in output
    assert "shards:" in output
    assert "/cost" in output
    assert "would have been" in output
