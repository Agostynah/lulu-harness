"""counterfactual.py: what query_all and flat_topk WOULD have cost, given
the same shards, computed WITHOUT re-running retrieval.

Every shard's CostProfile.cost_for(k) is known without contacting it --
that's exactly the property that makes the /cost command's headline
number possible on *every* turn, not just when evals/run.py happens to
execute a sweep. This is the mechanism behind the thesis's central
falsifiable claim (docs/THESIS.md): if these counterfactual numbers never
meaningfully exceed what routing actually spent, the sharding isn't
buying anything.
"""

from __future__ import annotations

from dataclasses import dataclass

from lulu_router.cost import Cost
from lulu_router.shard import Shard


@dataclass
class Counterfactual:
    label: str
    cost: Cost


def query_all_cost(shards: list[Shard], k: int) -> Cost:
    total = Cost()
    for shard in shards:
        total = total + shard.cost.cost_for(k)
    return total


def flat_topk_cost(shards: list[Shard], k: int) -> Cost:
    # A flat index touches every shard's underlying store exactly like
    # query_all does (see lulu_router.strategies.flat_topk's docstring):
    # a merged index has no notion of per-shard cost, so it cannot decline
    # to contact one. As a *cost* counterfactual the two are identical by
    # construction -- what evals/dbpedia measures as the actual difference
    # between the two baselines is recall behavior under a query-scoped
    # shard permission (see Shard.permits), not per-query spend.
    return query_all_cost(shards, k)


def compute_counterfactuals(shards: list[Shard], k: int) -> list[Counterfactual]:
    return [
        Counterfactual(label="query_all", cost=query_all_cost(shards, k)),
        Counterfactual(label="flat_topk", cost=flat_topk_cost(shards, k)),
    ]


def savings_pct(actual: Cost, baseline: Cost, attr: str = "tokens") -> float:
    """% saved on `attr` (tokens/latency_ms/usd) by `actual` relative to
    `baseline`. 0.0 for a zero-cost baseline (nothing to save against),
    not a division error."""
    baseline_value = getattr(baseline, attr)
    if baseline_value <= 0:
        return 0.0
    actual_value = getattr(actual, attr)
    return (1 - actual_value / baseline_value) * 100
