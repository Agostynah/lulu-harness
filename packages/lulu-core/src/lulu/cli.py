"""cli.py: the `lulu` command-line entrypoint.

Wires config -> ModelClient -> ToolRegistry -> PermissionChecker ->
AgentLoop -> Session into something actually runnable. This is what turns
everything else in this package from a tested library into a harness a
human can point at a project and use.

`model_override` exists specifically so tests can drive the whole wiring
end-to-end (argument parsing, tool registration, permission gating,
session logging) with a FakeModelClient instead of needing a real
Anthropic API key -- see tests/test_cli.py.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lulu.attention import AttentionMode
from lulu.config import VALID_PROVIDERS, LuluConfig, load_config
from lulu.llm.anthropic_client import AnthropicClient
from lulu.llm.client import Message, ModelClient
from lulu.llm.ollama_client import OllamaClient
from lulu.llm.openrouter_client import OpenRouterClient
from lulu.loop import AgentLoop
from lulu.permissions import PermissionChecker
from lulu.session import Session
from lulu.tools.bash_tool import make_bash_tool
from lulu.tools.file_tools import (
    make_edit_file_tool,
    make_glob_tool,
    make_grep_tool,
    make_read_file_tool,
    make_write_file_tool,
)
from lulu.tools.registry import ToolRegistry

SYSTEM_PROMPT = (
    "You are Lulu, an agentic coding assistant. You have file and shell "
    "tools scoped to the current project. Be direct and make the "
    "requested change."
)


def build_tool_registry(
    root: Path, locks_dir: Path | None = None, session_id: str | None = None
) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(make_read_file_tool(root))
    registry.register(make_write_file_tool(root, locks_dir=locks_dir, session_id=session_id))
    registry.register(make_edit_file_tool(root, locks_dir=locks_dir, session_id=session_id))
    registry.register(make_glob_tool(root))
    registry.register(make_grep_tool(root))
    registry.register(make_bash_tool(root))
    return registry


def build_model_client(config: LuluConfig) -> ModelClient:
    """Dispatches on config.provider (validated in config.py, so an
    unrecognized value never gets here). Each branch is a thin adapter --
    see llm/openai_compatible.py's docstring for why OpenRouter and Ollama
    share one implementation instead of duplicating it."""
    if config.provider == "openrouter":
        return OpenRouterClient(model=config.model, fallback_models=config.fallback_models)
    if config.provider == "ollama":
        return OllamaClient(model=config.model, fallback_models=config.fallback_models)
    return AnthropicClient(model=config.model, fallback_models=config.fallback_models)


def terminal_ask_human(tool_name: str, arguments: dict, reason: str) -> bool:
    print(f"\n[permission] {tool_name}({arguments}) -- {reason}")
    answer = input("Allow? [y/N] ").strip().lower()
    return answer == "y"


def _run_one_turn(loop: AgentLoop, session: Session, history: list[Message], user_input: str) -> list[Message]:
    messages_before = len(history)
    result = loop.run_turn(history, user_input)
    session.append_turn_result(result, messages_before)
    print(result.messages[-1].content)
    return result.messages


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lulu")
    parser.add_argument("prompt", nargs="?", help="One-shot prompt. Omit for an interactive session.")
    parser.add_argument("--root", default=".", help="Project root (default: cwd)")
    parser.add_argument(
        "--mode",
        choices=[m.value for m in AttentionMode],
        default=None,
        help="Override lulu.toml's attention_mode for this run",
    )
    parser.add_argument(
        "--provider",
        choices=VALID_PROVIDERS,
        default=None,
        help="Override lulu.toml's model.provider for this run",
    )
    parser.add_argument("--session", default=None, help="Resume an existing session id")
    return parser


def main(argv: list[str] | None = None, model_override: ModelClient | None = None) -> int:
    args = build_parser().parse_args(argv)

    root = Path(args.root).resolve()
    config = load_config(root / "lulu.toml")
    if args.provider:
        config.provider = args.provider
    mode = AttentionMode(args.mode) if args.mode else config.attention_mode
    log_dir = root / ".lulu" / "logs"
    locks_dir = root / ".lulu" / "locks"

    # Session first: its id is what write_file/edit_file claim locks
    # under and what PermissionChecker attributes those claims to, so
    # everything downstream needs it before it's built.
    session = (
        Session.resume(args.session, log_dir / "sessions")
        if args.session
        else Session.new(log_dir / "sessions")
    )
    history = session.load_history()

    model = model_override or build_model_client(config)
    tools = build_tool_registry(root, locks_dir=locks_dir, session_id=session.session_id)
    permissions = PermissionChecker(
        mode=mode,
        log_path=log_dir / "permissions.jsonl",
        locks_dir=locks_dir,
        session_id=session.session_id,
    )
    loop = AgentLoop(
        model=model,
        tools=tools,
        permissions=permissions,
        ask_human=terminal_ask_human,
        system=SYSTEM_PROMPT,
        max_iterations=config.max_iterations,
    )

    if args.prompt:
        _run_one_turn(loop, session, history, args.prompt)
        return 0

    print(f"Lulu -- session {session.session_id} -- mode={mode.value} -- root={root}")
    while True:
        try:
            user_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if user_input.strip() in ("exit", "quit"):
            return 0
        history = _run_one_turn(loop, session, history, user_input)


if __name__ == "__main__":
    sys.exit(main())
