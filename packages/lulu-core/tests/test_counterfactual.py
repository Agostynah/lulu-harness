"""counterfactual.py: pure arithmetic over shard cost profiles -- no
retrieval happens, so these tests never touch a real (or fake) store."""

from __future__ import annotations

from lulu_router.cost import Cost, CostProfile
from lulu_router.shard import Shard

from lulu.counterfactual import compute_counterfactuals, flat_topk_cost, query_all_cost, savings_pct


class _NullStore:
    def search(self, query_vec, k):
        raise AssertionError("counterfactual cost must never call search()")

    def __len__(self):
        return 0


def _shard(shard_id: str, latency_ms: float, usd: float, tokens_per_result: int) -> Shard:
    return Shard(
        id=shard_id,
        store=_NullStore(),
        cost=CostProfile(latency_ms=latency_ms, usd_per_query=usd, tokens_per_result=tokens_per_result),
    )


def test_query_all_cost_sums_every_shard():
    shards = [_shard("a", 5.0, 0.0, 80), _shard("b", 800.0, 0.002, 300)]

    cost = query_all_cost(shards, k=10)

    assert cost.latency_ms == 805.0
    assert cost.usd == 0.002
    assert cost.tokens == 800 + 3000


def test_query_all_cost_with_no_shards_is_zero():
    assert query_all_cost([], k=10) == Cost()


def test_flat_topk_cost_equals_query_all_cost():
    shards = [_shard("a", 5.0, 0.0, 80), _shard("b", 20.0, 0.0, 600)]
    assert flat_topk_cost(shards, k=5) == query_all_cost(shards, k=5)


def test_compute_counterfactuals_returns_both_baselines():
    shards = [_shard("a", 5.0, 0.0, 80)]

    results = compute_counterfactuals(shards, k=10)

    labels = {c.label for c in results}
    assert labels == {"query_all", "flat_topk"}


def test_savings_pct_computes_relative_reduction():
    actual = Cost(tokens=250)
    baseline = Cost(tokens=1000)
    assert savings_pct(actual, baseline, attr="tokens") == 75.0


def test_savings_pct_handles_zero_baseline_without_error():
    assert savings_pct(Cost(tokens=0), Cost(tokens=0), attr="tokens") == 0.0


def test_savings_pct_works_on_latency_and_usd_too():
    actual = Cost(latency_ms=10.0, usd=0.001)
    baseline = Cost(latency_ms=100.0, usd=0.004)
    assert savings_pct(actual, baseline, attr="latency_ms") == 90.0
    assert savings_pct(actual, baseline, attr="usd") == 75.0


def test_negative_savings_when_actual_exceeds_baseline():
    """Not clamped to zero -- a strategy that somehow costs MORE than the
    baseline should show as a negative saving, not be hidden."""
    actual = Cost(tokens=1500)
    baseline = Cost(tokens=1000)
    assert savings_pct(actual, baseline, attr="tokens") == -50.0
