"""load_config: missing file falls back to defaults entirely, partial
files merge with defaults field-by-field, and an invalid attention_mode
fails loudly with a clear message rather than silently picking something."""

from __future__ import annotations

from pathlib import Path

import pytest

from lulu.attention import AttentionMode
from lulu.config import LuluConfig, load_config


def test_missing_file_returns_all_defaults(tmp_path: Path):
    config = load_config(tmp_path / "lulu.toml")
    assert config == LuluConfig()


def test_full_config_is_parsed(tmp_path: Path):
    path = tmp_path / "lulu.toml"
    path.write_text(
        """
[model]
default = "claude-opus-5"
fallback = ["claude-sonnet-5", "claude-haiku-4-5-20251001"]

[harness]
attention_mode = "auto_edits"
max_iterations = 25
""",
        encoding="utf-8",
    )

    config = load_config(path)

    assert config.model == "claude-opus-5"
    assert config.fallback_models == ["claude-sonnet-5", "claude-haiku-4-5-20251001"]
    assert config.attention_mode == AttentionMode.AUTO_EDITS
    assert config.max_iterations == 25


def test_partial_config_merges_with_defaults(tmp_path: Path):
    path = tmp_path / "lulu.toml"
    path.write_text('[harness]\nattention_mode = "plan"\n', encoding="utf-8")

    config = load_config(path)

    assert config.attention_mode == AttentionMode.PLAN
    assert config.model == "claude-sonnet-5"  # default, untouched
    assert config.max_iterations == 50  # default, untouched


def test_empty_file_returns_defaults(tmp_path: Path):
    path = tmp_path / "lulu.toml"
    path.write_text("", encoding="utf-8")
    assert load_config(path) == LuluConfig()


def test_invalid_attention_mode_raises_clear_error(tmp_path: Path):
    path = tmp_path / "lulu.toml"
    path.write_text('[harness]\nattention_mode = "yolo_mode_typo"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid attention_mode"):
        load_config(path)
