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
from lulu.commands.render import render_cost, render_trace
from lulu.config import DEFAULT_FALLBACK_MODELS, DEFAULT_MODEL, VALID_PROVIDERS, LuluConfig, load_config
from lulu.llm.anthropic_client import AnthropicClient
from lulu.llm.anthropic_client import DEFAULT_MODEL as ANTHROPIC_DEFAULT_MODEL
from lulu.llm.client import Message, ModelClient
from lulu.llm.ollama_client import OllamaClient
from lulu.llm.ollama_client import DEFAULT_MODEL as OLLAMA_DEFAULT_MODEL
from lulu.llm.openrouter_client import OpenRouterClient
from lulu.llm.openrouter_client import DEFAULT_MODEL as OPENROUTER_DEFAULT_MODEL
from lulu.loop import AgentLoop
from lulu.memory import MemoryStore
from lulu.permissions import PermissionChecker
from lulu.profiles import DEFAULT_PROFILE_NAME, ProfileNotFoundError, load_profile
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
from lulu_router.trace import RoutingTrace


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


def _resolve_model(config: LuluConfig, provider_default: str) -> str:
    """config.model defaults to config.py's own DEFAULT_MODEL, which is
    Anthropic-style naming ("claude-sonnet-5") -- correct for provider
    "anthropic", wrong for the others (OpenRouter needs
    "anthropic/claude-sonnet-5", Ollama needs a locally-pulled tag like
    "llama3.1"). Caught by adversarial review, not assumed safe: passing
    config.model through unconditionally meant switching provider
    without ALSO hand-editing lulu.toml's model field silently sent the
    wrong model string to that provider's API. If the user never
    customized model away from the generic default, use the provider's
    OWN default instead; if they did customize it, respect that -- they
    presumably set something valid for their chosen provider."""
    if config.model == DEFAULT_MODEL:
        return provider_default
    return config.model


def _resolve_fallback_models(config: LuluConfig) -> list[str] | None:
    """Same bug, same fix, for the fallback chain: config.fallback_models
    defaults to ["claude-opus-5"] -- fine for Anthropic, an invalid slug
    for OpenRouter and a model tag Ollama was never told to pull for
    Ollama. If the user never customized it, pass None so each client
    falls back to its OWN default (OpenRouterClient/OllamaClient both
    default to no fallback chain at all, which is the honest "we don't
    know a safe cross-provider fallback" answer, not the Anthropic
    literal)."""
    if config.fallback_models == DEFAULT_FALLBACK_MODELS:
        return None
    return config.fallback_models


def build_model_client(config: LuluConfig) -> ModelClient:
    """Dispatches on config.provider (validated in config.py, so an
    unrecognized value never gets here). Each branch is a thin adapter --
    see llm/openai_compatible.py's docstring for why OpenRouter and Ollama
    share one implementation instead of duplicating it."""
    fallback_models = _resolve_fallback_models(config)
    if config.provider == "openrouter":
        model = _resolve_model(config, OPENROUTER_DEFAULT_MODEL)
        return OpenRouterClient(model=model, fallback_models=fallback_models)
    if config.provider == "ollama":
        model = _resolve_model(config, OLLAMA_DEFAULT_MODEL)
        return OllamaClient(model=model, fallback_models=fallback_models)
    model = _resolve_model(config, ANTHROPIC_DEFAULT_MODEL)
    return AnthropicClient(model=model, fallback_models=config.fallback_models)


def terminal_ask_human(tool_name: str, arguments: dict, reason: str) -> bool:
    print(f"\n[permission] {tool_name}({arguments}) -- {reason}")
    answer = input("Allow? [y/N] ").strip().lower()
    return answer == "y"


def _run_one_turn(
    loop: AgentLoop, session: Session, history: list[Message], user_input: str
) -> tuple[list[Message], RoutingTrace | None]:
    messages_before = len(history)
    result = loop.run_turn(history, user_input)
    session.append_turn_result(result, messages_before)
    print(result.messages[-1].content)
    return result.messages, result.trace


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
    parser.add_argument(
        "--scope",
        default=None,
        help="Identity this run is scoped to -- memory it writes/reads is boundaried "
        "to this scope (see Shard.permits in lulu_router). Omit for unscoped/personal use.",
    )
    parser.add_argument(
        "--profile",
        default=DEFAULT_PROFILE_NAME,
        help="Named persona to use for this run's system prompt (see profiles.py). "
        "Defaults to the built-in 'default' profile.",
    )
    return parser


def main(
    argv: list[str] | None = None,
    model_override: ModelClient | None = None,
    memory_override: MemoryStore | None = None,
) -> int:
    args = build_parser().parse_args(argv)

    root = Path(args.root).resolve()
    try:
        persona = load_profile(root, args.profile).persona
    except ProfileNotFoundError as exc:
        print(f"lulu: {exc}", file=sys.stderr)
        return 1
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
    memory = memory_override or MemoryStore()
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
        system=persona,
        max_iterations=config.max_iterations,
        memory=memory,
        memory_scope=args.scope,
    )

    if args.prompt:
        _run_one_turn(loop, session, history, args.prompt)
        return 0

    print(f"Lulu -- session {session.session_id} -- mode={mode.value} -- provider={config.provider} -- root={root}")
    last_trace: RoutingTrace | None = None
    while True:
        try:
            user_input = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        stripped = user_input.strip()
        if stripped in ("exit", "quit"):
            return 0
        if stripped == "/trace":
            print(render_trace(last_trace) if last_trace else "(no trace yet -- run a turn first)")
            continue
        if stripped == "/cost":
            if last_trace:
                print(render_cost(last_trace, memory.shards_for_scope(args.scope), memory.k))
            else:
                print("(no trace yet -- run a turn first)")
            continue
        history, last_trace = _run_one_turn(loop, session, history, user_input)


if __name__ == "__main__":
    sys.exit(main())
