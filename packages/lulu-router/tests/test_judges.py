"""Judge tests. GeometricJudge is tested end-to-end -- it's pure math, no
I/O. ClaudeCLIJudge is tested only at the prompt-building / output-parsing
boundary: the actual `claude -p` subprocess call is exercised manually via
evals/dbpedia/run.py (day 2's real deliverable), not on every `pytest` run,
so the suite stays fast and doesn't require the CLI to be installed in CI.
"""

from __future__ import annotations

from lulu_router.judges.claude_cli import ClaudeCLIJudge
from lulu_router.judges.fallback import FallbackJudge
from lulu_router.judges.geometric import GeometricJudge
from lulu_router.judges.jev import JevJudge
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


def test_jev_judge_parses_sufficient_answer():
    answer = {"answers": {"sufficient": {"type": "noul", "noul": 0.91}}}
    confidence, should_expand, reasoning = JevJudge._parse(answer)
    assert confidence == 0.91
    assert should_expand is False
    assert "0.910" in reasoning


def test_jev_judge_parses_insufficient_answer():
    answer = {"answers": {"sufficient": {"type": "noul", "noul": 0.2}}}
    confidence, should_expand, _reasoning = JevJudge._parse(answer)
    assert confidence == 0.2
    assert should_expand is True


def test_jev_judge_treats_unreachable_api_as_unconfident():
    confidence, should_expand, reasoning = JevJudge._parse(None)
    assert confidence == 0.0
    assert should_expand is True
    assert "unavailable" in reasoning


def test_jev_judge_handles_malformed_response():
    confidence, should_expand, _reasoning = JevJudge._parse({"answers": {}})
    assert confidence == 0.0
    assert should_expand is True


def test_jev_judge_builds_state_with_query_and_candidates():
    judge = JevJudge(api_key="fake-key")
    state = judge._build_state(
        "what is X", [_result(0.9, "candidate A")], sources_contacted=1, total_sources=3
    )
    assert "what is X" in state
    assert "candidate A" in state
    assert "1/3" in state


def test_claude_cli_judge_falls_back_when_cli_missing():
    judge = ClaudeCLIJudge(cli_path="this-binary-does-not-exist-anywhere")
    confidence, should_expand, reasoning = judge.judge("q", [_result(0.9)], 1, 3)
    assert confidence == 0.0
    assert should_expand is True
    assert "judge unavailable" in reasoning


class _StubJudge:
    """Minimal stand-in for exercising FallbackJudge without a real Jev
    call: returns a fixed verdict and lets the test flip `available`."""

    def __init__(self, name: str, confidence: float, should_expand: bool, available: bool = True):
        self.name = name
        self._verdict = (confidence, should_expand, f"{name}: verdict")
        self.available = available

    def judge(self, query, results, sources_contacted, total_sources):
        return self._verdict


def test_fallback_judge_uses_primary_when_available():
    primary = _StubJudge("primary", confidence=0.9, should_expand=False, available=True)
    secondary = _StubJudge("secondary", confidence=0.1, should_expand=True)
    judge = FallbackJudge(primary=primary, secondary=secondary)
    confidence, should_expand, reasoning = judge.judge("q", [_result(0.9)], 1, 3)
    assert confidence == 0.9
    assert should_expand is False
    assert "secondary" not in reasoning


def test_fallback_judge_falls_back_when_primary_unavailable():
    primary = _StubJudge("primary", confidence=0.0, should_expand=True, available=False)
    secondary = _StubJudge("secondary", confidence=0.8, should_expand=False)
    judge = FallbackJudge(primary=primary, secondary=secondary)
    confidence, should_expand, reasoning = judge.judge("q", [_result(0.9)], 1, 3)
    assert confidence == 0.8
    assert should_expand is False
    assert "fell back" in reasoning


def test_fallback_judge_does_not_fall_back_on_a_real_low_confidence_verdict():
    # A genuine low-confidence answer from the primary is a real verdict,
    # not a failure -- must NOT trigger the fallback.
    primary = _StubJudge("primary", confidence=0.15, should_expand=True, available=True)
    secondary = _StubJudge("secondary", confidence=0.99, should_expand=False)
    judge = FallbackJudge(primary=primary, secondary=secondary)
    confidence, should_expand, _reasoning = judge.judge("q", [_result(0.9)], 1, 3)
    assert confidence == 0.15
    assert should_expand is True


def test_fallback_judge_name_combines_both():
    judge = FallbackJudge(primary=GeometricJudge(), secondary=GeometricJudge())
    assert judge.name == "geometric+geometric"


def test_jev_judge_marks_unavailable_after_a_failed_call(monkeypatch):
    judge = JevJudge(api_key="fake-key")
    monkeypatch.setattr(judge, "_call_api", lambda state: None)
    judge.judge("q", [_result(0.9)], 1, 3)
    assert judge.available is False


def test_jev_judge_marks_available_after_a_real_verdict(monkeypatch):
    judge = JevJudge(api_key="fake-key")
    monkeypatch.setattr(judge, "_call_api", lambda state: {"answers": {"sufficient": {"noul": 0.7}}})
    judge.judge("q", [_result(0.9)], 1, 3)
    assert judge.available is True
