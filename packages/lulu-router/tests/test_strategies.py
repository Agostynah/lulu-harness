"""Behavioral tests for each of the six strategies, run against the
synthetic 3-cluster corpus from conftest.py. These assert on *which shards
get contacted and why*, not on retrieval quality (that's evals/dbpedia's
job, day 2) -- the point here is that the routing logic itself does what
the paper's strategy descriptions say it does.
"""

from __future__ import annotations

from lulu_router.cost import Budget
from lulu_router.strategies import (
    STRATEGIES,
    budgeted_communication,
    confidence_threshold,
    flat_topk,
    progressive_expansion,
    query_all,
    top_n_neighbors,
)
from .conftest import query_vec_for_cluster


class NeverSatisfiedJudge:
    """A judge that always reports low confidence and always wants to
    expand -- used to force strategies through their full expansion path
    regardless of how easy the query actually is."""

    name = "never_satisfied"

    def judge(self, query, results, sources_contacted, total_sources):
        return 0.1, sources_contacted < total_sources, "never enough"


def test_query_all_contacts_every_shard(shards, budget, judge, synthetic_corpus):
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    trace = query_all("q", qvec, shards, budget, k=5, judge=judge)
    assert set(trace.shards_contacted) == {s.id for s in shards}
    assert trace.shards_skipped == []


def test_flat_topk_also_contacts_every_shard(shards, budget, judge, synthetic_corpus):
    """flat_topk cannot skip a shard by construction -- a merged index has
    no per-shard cost to weigh, so it always touches everything. That
    inability is the capability gap the thesis rests on, not a bug in this
    baseline."""
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    trace = flat_topk("q", qvec, shards, budget, k=5, judge=judge)
    assert trace.strategy == "flat_topk"
    assert set(trace.shards_contacted) == {s.id for s in shards}


def test_top_n_neighbors_respects_n(shards, budget, judge, synthetic_corpus):
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    trace = top_n_neighbors("q", qvec, shards, budget, k=5, judge=judge, n=1)
    assert len(trace.shards_contacted) == 1
    # the one contacted shard should be the one whose centroid the query
    # is actually closest to
    best = max(trace.shards_considered, key=lambda s: s.centroid_similarity)
    assert trace.shards_contacted[0] == best.shard_id


def test_top_n_neighbors_marks_the_rest_outside_top_n(shards, budget, judge, synthetic_corpus):
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    trace = top_n_neighbors("q", qvec, shards, budget, k=5, judge=judge, n=1)
    skipped = trace.shards_skipped
    assert len(skipped) == len(shards) - 1
    assert all(s.skip_reason == "outside_top_n" for s in skipped)


def test_confidence_threshold_stops_early_on_high_confidence(shards, budget, judge, synthetic_corpus):
    """A query dead-center in one cluster should hit high confidence on the
    first (closest) shard and never need to expand."""
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    trace = confidence_threshold("q", qvec, shards, budget, k=5, judge=judge, tau=0.5)
    assert len(trace.shards_contacted) == 1
    assert trace.confidence >= 0.5


def test_confidence_threshold_expands_when_never_satisfied(shards, budget, synthetic_corpus):
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    trace = confidence_threshold(
        "q", qvec, shards, budget, k=5, judge=NeverSatisfiedJudge(), tau=0.99
    )
    assert set(trace.shards_contacted) == {s.id for s in shards}


def test_budgeted_communication_respects_hard_cap(shards, judge, synthetic_corpus, expensive_cost):
    """A budget that can't afford even the first (cheapest still-too-costly)
    shard should leave every shard uncontacted -- the cap is a hard stop,
    not a soft preference the strategy can violate once."""
    for shard in shards:
        shard.cost = expensive_cost
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    tiny_budget = Budget(max_tokens=10, max_latency_ms=1.0, max_usd=0.0)
    trace = budgeted_communication("q", qvec, shards, tiny_budget, k=5, judge=judge)
    assert trace.shards_contacted == []
    assert all(s.skip_reason == "budget_exhausted" for s in trace.shards_skipped)


def test_progressive_expansion_stops_at_budget_even_if_judge_wants_more(
    shards, synthetic_corpus, expensive_cost
):
    """A budget that affords exactly one expensive shard should stop
    expansion there, even against a judge that always asks for more."""
    for shard in shards:
        shard.cost = expensive_cost
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    one_shard_budget = Budget(
        max_tokens=expensive_cost.tokens_per_result * 5 + 1,
        max_latency_ms=5000.0,
        max_usd=1.0,
    )
    trace = progressive_expansion(
        "q", qvec, shards, one_shard_budget, k=5, judge=NeverSatisfiedJudge(), tau=0.99
    )
    assert len(trace.shards_contacted) == 1
    assert all(
        s.skip_reason == "budget_exhausted" for s in trace.shards_skipped
    )


def test_results_are_ranked_by_score(shards, budget, judge, synthetic_corpus):
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    trace = query_all("q", qvec, shards, budget, k=10, judge=judge)
    scores = [r.score for r in trace.results]
    assert scores == sorted(scores, reverse=True)


def test_denied_scope_excludes_shard_entirely(shards, budget, judge, synthetic_corpus):
    """A shard the caller's scope doesn't permit should never appear in
    shards_considered at all -- access control is a hard boundary a flat
    index has no way to express, so the trace shouldn't blur it together
    with an ordinary cost/confidence skip."""
    shards[0].allowed_scopes = frozenset({"customer-a"})
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    trace = query_all("q", qvec, shards, budget, k=5, judge=judge, scope="customer-b")
    considered_ids = {s.shard_id for s in trace.shards_considered}
    assert shards[0].id not in considered_ids
    assert set(trace.shards_contacted) == {s.id for s in shards[1:]}


def test_all_strategies_registered():
    expected = {
        "query_all",
        "flat_topk",
        "top_n_neighbors",
        "confidence_threshold",
        "progressive_expansion",
        "budgeted_communication",
    }
    assert set(STRATEGIES) == expected
