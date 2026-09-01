"""config.py: lulu.toml loading, plus a minimal .env loader.

Minimal by design -- just enough to pick the model provider, model,
fallback chain, default AttentionMode, and iteration cap. Every value has
a sane default, so a project with no lulu.toml at all still runs.

The .env loading exists because it was missing entirely: .env.example's
own comment claimed "the Anthropic SDK picks this up automatically",
which is only true if something has ALREADY exported the variable into
the process's real environment -- nothing in this codebase ever read the
.env file itself before this. Caught while wiring up an onboarding
wizard that writes a key to .env; a write nothing reads back is a
feature that looks like it works and doesn't. No new dependency
(python-dotenv) for four lines of KEY=VALUE parsing -- os.environ.setdefault
so a real shell-exported variable always wins over the file, standard
dotenv precedence.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from lulu.attention import AttentionMode

DEFAULT_PROVIDER = "anthropic"
VALID_PROVIDERS = ("anthropic", "openrouter", "ollama")
DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_FALLBACK_MODELS = ["claude-opus-5"]
DEFAULT_MAX_ITERATIONS = 50


@dataclass
class LuluConfig:
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL
    fallback_models: list[str] = field(default_factory=lambda: list(DEFAULT_FALLBACK_MODELS))
    attention_mode: AttentionMode = AttentionMode.MANUAL
    max_iterations: int = DEFAULT_MAX_ITERATIONS


def load_dotenv(root: Path) -> None:
    """Reads root/.env (if it exists) and sets any KEY=VALUE line into
    os.environ, skipping blank lines and #-comments. Uses setdefault, not
    a plain assignment, so a variable the shell already exported takes
    precedence over the file -- the file is a fallback, not an override."""
    env_path = root / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def write_env_var(root: Path, key: str, value: str) -> None:
    """Sets KEY=VALUE in root/.env, replacing an existing line for that
    key if present, appending otherwise. Used by the onboarding wizard
    (server.py's /api/apikey) so "paste your key" actually persists
    somewhere load_dotenv() will pick up on the next run -- not just
    os.environ for the life of this process. Read-modify-write on the
    whole file rather than a real key-value store; .env is meant to be a
    handful of lines a person might also open and edit by hand."""
    env_path = root / ".env"
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []

    new_line = f"{key}={value}"
    replaced = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or "=" not in stripped:
            continue
        existing_key = stripped.partition("=")[0].strip()
        if existing_key == key:
            lines[i] = new_line
            replaced = True
            break
    if not replaced:
        lines.append(new_line)

    # newline="": same reasoning as tools/file_tools.py -- don't let a
    # write silently normalize line endings in a file a person may
    # already have open.
    with env_path.open("w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")


def load_config(path: Path) -> LuluConfig:
    load_dotenv(path.parent)

    if not path.exists():
        return LuluConfig()

    with path.open("rb") as f:
        data = tomllib.load(f)

    model_section = data.get("model", {})
    harness_section = data.get("harness", {})

    provider = model_section.get("provider", DEFAULT_PROVIDER)
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"invalid model.provider in lulu.toml: {provider!r}; must be one of {VALID_PROVIDERS}")

    mode_str = harness_section.get("attention_mode", AttentionMode.MANUAL.value)
    try:
        mode = AttentionMode(mode_str)
    except ValueError:
        valid = [m.value for m in AttentionMode]
        raise ValueError(f"invalid attention_mode in lulu.toml: {mode_str!r}; must be one of {valid}") from None

    return LuluConfig(
        provider=provider,
        model=model_section.get("default", DEFAULT_MODEL),
        fallback_models=list(model_section.get("fallback", DEFAULT_FALLBACK_MODELS)),
        attention_mode=mode,
        max_iterations=harness_section.get("max_iterations", DEFAULT_MAX_ITERATIONS),
    )
