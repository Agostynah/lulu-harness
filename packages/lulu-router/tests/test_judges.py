"""Judge tests. GeometricJudge is tested end-to-end -- it's pure math, no
I/O. ClaudeCLIJudge is tested only at the prompt-building / output-parsing
boundary: the actual `claude -p` subprocess call is exercised manually via
evals/dbpedia/run.py (day 2's real deliverable), not on every `pytest` run,
so the suite stays fast and doesn't require the CLI to be installed in CI.
"""

from __future__ import annotations

from lulu_router.judges.claude_cli import ClaudeCLIJudge
from lulu_router.judges.geometric import GeometricJudge
from lulu_router.shard import SearchResult


def _result(score: float, content: str = "x") -> SearchResult:
    return SearchResult(id="r", content=content, score=score)


def test_geometric_judge_high_gap_full_coverage_is_confident():
    judge = GeometricJudge(threshold=0.6)
    confidence, should_expand, _ = judge.judge(
        "q", [_result(0.95), _result(0.10)], sources_contacted=3, total_sources=3
    )
    assert confidence > 0.6
    assert should_expand is False


def test_geometric_judge_low_gap_is_unconfident():
    judge = GeometricJudge(threshold=0.6)
    confidence, should_expand, _ = judge.judge(
        "q", [_result(0.51), _result(0.50)], sources_contacted=3, total_sources=3
    )
    assert confidence < 0.6
    assert should_expand is True


def test_geometric_judge_partial_coverage_reduces_confidence():
    judge = GeometricJudge(threshold=0.6)
    full, _, _ = judge.judge("q", [_result(0.95), _result(0.10)], 3, 3)
    partial, _, _ = judge.judge("q", [_result(0.95), _result(0.10)], 1, 3)
    assert partial < full


def test_geometric_judge_no_results_is_zero_confidence():
    judge = GeometricJudge()
    confidence, should_expand, _ = judge.judge("q", [], 0, 3)
    assert confidence == 0.0
    assert should_expand is True


def test_claude_cli_judge_parses_clean_json():
    raw = '{"sufficient": true, "confidence": 0.87, "missing": ""}'
    confidence, should_expand, _reasoning = ClaudeCLIJudge._parse(raw)
    assert confidence == 0.87
    assert should_expand is False


def test_claude_cli_judge_parses_json_wrapped_in_prose():
    raw = (
        "Sure, here is my judgement:\n"
        '{"sufficient": false, "confidence": 0.2, "missing": "no code-related results"}\n'
        "Hope that helps."
    )
    confidence, should_expand, reasoning = ClaudeCLIJudge._parse(raw)
    assert confidence == 0.2
    assert should_expand is True
    assert "no code-related results" in reasoning


def test_claude_cli_judge_handles_unparseable_output():
    confidence, should_expand, _reasoning = ClaudeCLIJudge._parse("not json at all")
    assert confidence == 0.0
    assert should_expand is True


def test_claude_cli_judge_handles_malformed_json():
    confidence, should_expand, _reasoning = ClaudeCLIJudge._parse('{"sufficient": true, "confidence":')
    assert confidence == 0.0
    assert should_expand is True


def test_claude_cli_judge_builds_prompt_with_substitutions(tmp_path, monkeypatch):
    prompt_file = tmp_path / "sufficiency.md"
    prompt_file.write_text(
        "Q: {{query}} | {{sources_contacted}}/{{total_sources}} | {{candidates}}",
        encoding="utf-8",
    )
    monkeypatch.setattr("lulu_router.judges.claude_cli.PROMPT_PATH", prompt_file)

    judge = ClaudeCLIJudge()
    prompt = judge._build_prompt(
        "what is X", [_result(0.9, "candidate A")], sources_contacted=1, total_sources=3
    )
    assert "what is X" in prompt
    assert "1/3" in prompt
    assert "candidate A" in prompt


def test_claude_cli_judge_falls_back_when_cli_missing():
    judge = ClaudeCLIJudge(cli_path="this-binary-does-not-exist-anywhere")
    confidence, should_expand, reasoning = judge.judge("q", [_result(0.9)], 1, 3)
    assert confidence == 0.0
    assert should_expand is True
    assert "judge unavailable" in reasoning
