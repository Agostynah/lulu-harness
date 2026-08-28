"""suggest_promotions(): streak counting over permissions.jsonl, reset on
denial, and never suggesting a tool that's already ALLOW under the
current mode. No ML -- pure aggregation, so these are pure log-shape
tests."""

from __future__ import annotations

import json
from pathlib import Path


from lulu.attention import AttentionMode, Decision, suggest_promotions


def _write_log(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _approved(tool: str) -> dict:
    return {"tool": tool, "outcome": "approved"}


def _denied(tool: str) -> dict:
    return {"tool": tool, "outcome": "denied"}


def test_missing_log_returns_no_suggestions(tmp_path: Path):
    assert suggest_promotions(tmp_path / "nope.jsonl") == []


def test_no_suggestion_below_min_streak(tmp_path: Path):
    log = tmp_path / "permissions.jsonl"
    _write_log(log, [_approved("write_file")] * 3)
    assert suggest_promotions(log, min_streak=5) == []


def test_suggests_promotion_at_streak_threshold(tmp_path: Path):
    log = tmp_path / "permissions.jsonl"
    _write_log(log, [_approved("write_file")] * 5)

    suggestions = suggest_promotions(log, min_streak=5, mode=AttentionMode.MANUAL)

    assert len(suggestions) == 1
    assert suggestions[0].tool == "write_file"
    assert suggestions[0].approval_streak == 5
    assert suggestions[0].current_decision == Decision.ASK


def test_denial_resets_the_streak(tmp_path: Path):
    log = tmp_path / "permissions.jsonl"
    _write_log(log, [_approved("write_file")] * 4 + [_denied("write_file")] + [_approved("write_file")] * 2)

    suggestions = suggest_promotions(log, min_streak=5)

    assert suggestions == []  # only 2 approvals since the denial


def test_streak_after_denial_can_still_reach_threshold(tmp_path: Path):
    log = tmp_path / "permissions.jsonl"
    _write_log(log, [_approved("write_file")] * 4 + [_denied("write_file")] + [_approved("write_file")] * 5)

    suggestions = suggest_promotions(log, min_streak=5)

    assert len(suggestions) == 1
    assert suggestions[0].approval_streak == 5


def test_already_allowed_tool_is_not_suggested(tmp_path: Path):
    """In AUTO mode, write_file is already ALLOW -- nothing to suggest."""
    log = tmp_path / "permissions.jsonl"
    _write_log(log, [_approved("write_file")] * 10)

    suggestions = suggest_promotions(log, min_streak=5, mode=AttentionMode.AUTO)

    assert suggestions == []


def test_multiple_tools_tracked_independently(tmp_path: Path):
    log = tmp_path / "permissions.jsonl"
    _write_log(
        log,
        [_approved("write_file")] * 5 + [_approved("bash")] * 2,
    )

    suggestions = suggest_promotions(log, min_streak=5, mode=AttentionMode.MANUAL)

    tools = {s.tool for s in suggestions}
    assert tools == {"write_file"}


def test_records_missing_tool_or_outcome_are_ignored(tmp_path: Path):
    log = tmp_path / "permissions.jsonl"
    _write_log(log, [{"something_else": True}] + [_approved("write_file")] * 5)

    suggestions = suggest_promotions(log, min_streak=5)

    assert len(suggestions) == 1


def test_blank_lines_in_log_are_ignored(tmp_path: Path):
    log = tmp_path / "permissions.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(_approved("write_file")) for _ in range(5)]
    log.write_text("\n".join(lines) + "\n\n\n", encoding="utf-8")

    suggestions = suggest_promotions(log, min_streak=5)

    assert len(suggestions) == 1
