"""load_config: missing file falls back to defaults entirely, partial
files merge with defaults field-by-field, and an invalid attention_mode
fails loudly with a clear message rather than silently picking something."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from lulu.attention import AttentionMode
from lulu.config import LuluConfig, load_config, load_dotenv, write_env_var


def test_load_dotenv_with_no_env_file_does_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("SOME_TEST_KEY", raising=False)
    load_dotenv(tmp_path)
    assert "SOME_TEST_KEY" not in os.environ


def test_load_dotenv_sets_variables_from_the_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LULU_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text("LULU_TEST_KEY=hello-world\n", encoding="utf-8")

    load_dotenv(tmp_path)

    assert os.environ["LULU_TEST_KEY"] == "hello-world"


def test_load_dotenv_skips_blank_lines_and_comments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LULU_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text("\n# a comment\nLULU_TEST_KEY=value\n\n", encoding="utf-8")

    load_dotenv(tmp_path)

    assert os.environ["LULU_TEST_KEY"] == "value"


def test_load_dotenv_strips_surrounding_quotes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LULU_TEST_KEY", raising=False)
    (tmp_path / ".env").write_text('LULU_TEST_KEY="quoted-value"\n', encoding="utf-8")

    load_dotenv(tmp_path)

    assert os.environ["LULU_TEST_KEY"] == "quoted-value"


def test_load_dotenv_never_overrides_an_already_exported_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("LULU_TEST_KEY", "from-shell")
    (tmp_path / ".env").write_text("LULU_TEST_KEY=from-dotenv\n", encoding="utf-8")

    load_dotenv(tmp_path)

    assert os.environ["LULU_TEST_KEY"] == "from-shell"


def test_write_env_var_creates_the_file_when_missing(tmp_path: Path):
    write_env_var(tmp_path, "ANTHROPIC_API_KEY", "sk-test-123")
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "ANTHROPIC_API_KEY=sk-test-123\n"


def test_write_env_var_appends_to_an_existing_file(tmp_path: Path):
    (tmp_path / ".env").write_text("OTHER_KEY=unrelated\n", encoding="utf-8")

    write_env_var(tmp_path, "ANTHROPIC_API_KEY", "sk-test-123")

    text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "OTHER_KEY=unrelated" in text
    assert "ANTHROPIC_API_KEY=sk-test-123" in text


def test_write_env_var_replaces_an_existing_value_for_the_same_key(tmp_path: Path):
    (tmp_path / ".env").write_text("ANTHROPIC_API_KEY=old-value\nOTHER_KEY=unrelated\n", encoding="utf-8")

    write_env_var(tmp_path, "ANTHROPIC_API_KEY", "new-value")

    lines = (tmp_path / ".env").read_text(encoding="utf-8").splitlines()
    assert "ANTHROPIC_API_KEY=new-value" in lines
    assert "ANTHROPIC_API_KEY=old-value" not in lines
    assert "OTHER_KEY=unrelated" in lines
    assert len(lines) == 2  # replaced in place, not duplicated


def test_write_env_var_then_load_dotenv_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("LULU_TEST_KEY", raising=False)
    write_env_var(tmp_path, "LULU_TEST_KEY", "round-tripped")
    load_dotenv(tmp_path)
    assert os.environ["LULU_TEST_KEY"] == "round-tripped"


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


def test_default_provider_is_anthropic(tmp_path: Path):
    assert load_config(tmp_path / "lulu.toml").provider == "anthropic"


def test_provider_is_parsed(tmp_path: Path):
    path = tmp_path / "lulu.toml"
    path.write_text('[model]\nprovider = "openrouter"\n', encoding="utf-8")

    config = load_config(path)

    assert config.provider == "openrouter"
    assert config.model == "claude-sonnet-5"  # default, untouched


def test_ollama_provider_is_valid(tmp_path: Path):
    path = tmp_path / "lulu.toml"
    path.write_text('[model]\nprovider = "ollama"\n', encoding="utf-8")
    assert load_config(path).provider == "ollama"


def test_invalid_provider_raises_clear_error(tmp_path: Path):
    path = tmp_path / "lulu.toml"
    path.write_text('[model]\nprovider = "some_typo_provider"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="invalid model.provider"):
        load_config(path)
