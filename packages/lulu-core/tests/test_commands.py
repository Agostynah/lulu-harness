"""commands/loader.py and commands/render.py: the loader is a thin
filesystem lookup, tested for the missing-dir and happy-path cases; the
renderers are tested for structural content (they compose
RoutingTrace + counterfactual, both already independently tested) rather
than pinned to exact ASCII-art formatting."""

from __future__ import annotations

from pathlib import Path

from lulu_router.cost import Budget, Cost, CostProfile
from lulu_router.shard import Shard, SearchResult
from lulu_router.trace import ExpansionRound, RoutingTrace, ShardScore

from lulu.commands.loader import load_commands
from lulu.commands.render import render_cost, render_trace


class _NullStore:
    def search(self, query_vec, k):
        return []

    def __len__(self):
        return 0


def _shard(shard_id: str) -> Shard:
    return Shard(id=shard_id, store=_NullStore(), cost=CostProfile(5.0, 0.0, 80))


# --- loader ---


def test_load_commands_on_missing_dir_returns_empty(tmp_path: Path):
    assert load_commands(tmp_path / "nope") == {}


def test_load_commands_reads_markdown_files(tmp_path: Path):
    commands_dir = tmp_path / "commands"
    commands_dir.mkdir()
    (commands_dir / "mem.md").write_text("# /mem\nsearch memory", encoding="utf-8")
    (commands_dir / "shards.md").write_text("# /shards\nlist shards", encoding="utf-8")
    (commands_dir / "not_a_command.txt").write_text("ignored", encoding="utf-8")

    commands = load_commands(commands_dir)

    assert set(commands) == {"mem", "shards"}
    assert commands["mem"].body == "# /mem\nsearch memory"


# --- render_trace ---


def _sample_trace() -> RoutingTrace:
    return RoutingTrace(
        query="what did we decide",
        strategy="progressive_expansion",
        judge="geometric",
        budget=Budget(),
        shards_considered=[
            ShardScore("episodic", centroid_similarity=0.91, contacted=True),
            ShardScore("code", centroid_similarity=0.21, contacted=False, skip_reason="confidence_met"),
        ],
        rounds=[
            ExpansionRound(
                round_index=0,
                shard_id="episodic",
                confidence_before=0.0,
                confidence_after=0.87,
                judge="geometric",
                verdict="sufficient",
                reasoning="gap=0.8 coverage=1.0",
            )
        ],
        confidence=0.87,
        spent=Cost(latency_ms=45.0, usd=0.0, tokens=1240),
        results=[SearchResult(id="e1", content="decided to use SQLite", score=0.91)],
    )


def test_render_trace_shows_contacted_and_skipped_shards():
    output = render_trace(_sample_trace())
    assert "✓ episodic" in output
    assert "✗ code" in output
    assert "confidence_met" in output


def test_render_trace_shows_expansion_rounds():
    output = render_trace(_sample_trace())
    assert "0.87" in output
    assert "sufficient" in output


def test_render_trace_shows_spend():
    output = render_trace(_sample_trace())
    assert "1240 tok" in output
    assert "$0.0000" in output


# --- render_cost ---


def test_render_cost_shows_shard_marks():
    trace = _sample_trace()
    shards = [_shard("episodic"), _shard("code")]
    output = render_cost(trace, shards, k=10)
    assert "episodic ✓" in output
    assert "code ✗(confidence_met)" in output


def test_render_cost_shows_counterfactual_savings():
    trace = _sample_trace()
    shards = [_shard("episodic"), _shard("code")]
    output = render_cost(trace, shards, k=10)
    assert "query_all would have been:" in output
    assert "flat_topk would have been:" in output
    assert "%" in output


def test_render_cost_savings_are_positive_when_routing_spent_less():
    """The sample trace spent 1240 tok; query_all over 2 shards at k=10
    with tokens_per_result=80 each would be 1600 tok -- routing should
    show as a positive percentage saved (a '+' sign), not negative."""
    trace = _sample_trace()
    shards = [_shard("episodic"), _shard("code")]
    output = render_cost(trace, shards, k=10)
    query_all_line = next(line for line in output.splitlines() if line.startswith("query_all"))
    assert "+" in query_all_line
    assert "-" not in query_all_line.split("->")[1]
