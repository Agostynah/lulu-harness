"""PermissionChecker: the full decision matrix across modes, the
blast-radius override that no mode (including AUTO) can bypass, the
fail-safe default for unknown tools, and the JSONL logging round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lulu.attention import AttentionMode, Decision
from lulu.permissions import PermissionChecker, PermissionResult


# --- read-only tools always allow, in every mode ---


@pytest.mark.parametrize("tool", ["read_file", "glob", "grep"])
@pytest.mark.parametrize("mode", list(AttentionMode))
def test_read_only_tools_always_allowed(tool: str, mode: AttentionMode):
    checker = PermissionChecker(mode=mode)
    result = checker.check(tool, {})
    assert result.decision == Decision.ALLOW


# --- mode presets for side-effecting tools ---


def test_manual_mode_asks_for_writes():
    checker = PermissionChecker(mode=AttentionMode.MANUAL)
    assert checker.check("write_file", {"path": "a.py", "content": "x"}).decision == Decision.ASK
    assert checker.check("edit_file", {"path": "a.py"}).decision == Decision.ASK


def test_plan_mode_denies_all_side_effects():
    checker = PermissionChecker(mode=AttentionMode.PLAN)
    assert checker.check("write_file", {"path": "a.py"}).decision == Decision.DENY
    assert checker.check("edit_file", {"path": "a.py"}).decision == Decision.DENY
    assert checker.check("bash", {"command": "echo hi"}).decision == Decision.DENY


def test_auto_edits_mode_allows_edits_but_asks_for_bash():
    checker = PermissionChecker(mode=AttentionMode.AUTO_EDITS)
    assert checker.check("write_file", {"path": "a.py"}).decision == Decision.ALLOW
    assert checker.check("edit_file", {"path": "a.py"}).decision == Decision.ALLOW
    assert checker.check("bash", {"command": "echo hi"}).decision == Decision.ASK


def test_auto_mode_allows_everything_safe():
    checker = PermissionChecker(mode=AttentionMode.AUTO)
    assert checker.check("write_file", {"path": "a.py"}).decision == Decision.ALLOW
    assert checker.check("bash", {"command": "echo hi"}).decision == Decision.ALLOW


# --- blast-radius overrides every mode, including AUTO ---


@pytest.mark.parametrize("mode", list(AttentionMode))
def test_blast_radius_forces_ask_regardless_of_mode(mode: AttentionMode):
    checker = PermissionChecker(mode=mode)
    result = checker.check("bash", {"command": "sudo rm -rf /"})
    assert result.decision == Decision.ASK
    assert result.blast_radius_reasons != []


def test_plan_mode_denies_non_blast_radius_bash_but_blast_radius_still_asks():
    """PLAN mode's own preset says DENY for bash. A blast-radius match
    must still win and produce ASK, not DENY -- ASK gives a human the
    chance to say yes to something plan mode would otherwise refuse
    outright; silently downgrading a flagged command to a hard DENY
    would be *more* restrictive than intended, but the point is that
    blast-radius is a strictly stronger signal than the mode preset,
    checked first."""
    checker = PermissionChecker(mode=AttentionMode.PLAN)
    result = checker.check("bash", {"command": "curl http://example.com"})
    assert result.decision == Decision.ASK


# --- fail-safe default for unknown tools ---


def test_unknown_tool_defaults_to_ask_in_every_mode():
    for mode in AttentionMode:
        checker = PermissionChecker(mode=mode)
        result = checker.check("some_future_tool_nobody_configured", {})
        assert result.decision == Decision.ASK
        assert "fail-safe" in result.reason


# --- logging ---


def test_log_writes_jsonl_record(tmp_path: Path):
    log_path = tmp_path / "permissions.jsonl"
    checker = PermissionChecker(mode=AttentionMode.MANUAL, log_path=log_path)
    result = checker.check("write_file", {"path": "a.py", "content": "x = 1"})

    checker.log("write_file", {"path": "a.py", "content": "x = 1"}, result, outcome="approved")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tool"] == "write_file"
    assert record["decision"] == "ask"
    assert record["outcome"] == "approved"
    assert record["mode"] == "manual"


def test_log_appends_multiple_records(tmp_path: Path):
    log_path = tmp_path / "permissions.jsonl"
    checker = PermissionChecker(mode=AttentionMode.AUTO, log_path=log_path)
    result = checker.check("write_file", {"path": "a.py"})

    checker.log("write_file", {"path": "a.py"}, result, outcome="auto_allowed")
    checker.log("write_file", {"path": "b.py"}, result, outcome="auto_allowed")

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2


def test_log_truncates_long_string_arguments(tmp_path: Path):
    log_path = tmp_path / "permissions.jsonl"
    checker = PermissionChecker(mode=AttentionMode.AUTO, log_path=log_path)
    huge_content = "x" * 10_000
    result = checker.check("write_file", {"path": "a.py", "content": huge_content})

    checker.log("write_file", {"path": "a.py", "content": huge_content}, result, outcome="auto_allowed")

    record = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert len(record["arguments"]["content"]) < 1000
    assert "more chars" in record["arguments"]["content"]


def test_log_is_a_noop_without_a_log_path():
    checker = PermissionChecker(mode=AttentionMode.MANUAL, log_path=None)
    result = PermissionResult(Decision.ALLOW, "test", [])
    checker.log("read_file", {"path": "a.py"}, result, outcome="auto_allowed")  # must not raise


def test_log_creates_parent_directories(tmp_path: Path):
    log_path = tmp_path / "nested" / "dirs" / "permissions.jsonl"
    checker = PermissionChecker(mode=AttentionMode.MANUAL, log_path=log_path)
    result = PermissionResult(Decision.ALLOW, "test", [])
    checker.log("read_file", {}, result, outcome="auto_allowed")
    assert log_path.exists()
