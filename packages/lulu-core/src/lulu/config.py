"""config.py: lulu.toml loading.

Minimal by design -- just enough to pick the model provider, model,
fallback chain, default AttentionMode, and iteration cap. Every value has
a sane default, so a project with no lulu.toml at all still runs.
"""

from __future__ import annotations

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


def load_config(path: Path) -> LuluConfig:
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
