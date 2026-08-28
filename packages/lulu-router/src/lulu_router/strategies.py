"""The six routing strategies under test.

Two baselines:
  - `query_all`  -- contact every permitted shard (the paper's global
    baseline: maximum recall, maximum cost).
  - `flat_topk`  -- pretend all shards are one merged index and take top-k
    globally. This is the skeptic's baseline: "why shard at all, just use
    one index." A flat index has no notion of per-shard cost, so by
    construction it cannot decline to contact a shard -- that inability is
    exactly the capability gap this project's thesis rests on, not a
    handicap imposed on the baseline.

Four from the paper (distributed-vector-memory-routing), each parameterized
by a Judge that decides when accumulated confidence is "enough":
  - `top_n_neighbors`         -- rank shards by centroid similarity, always
    contact the closest N, no adaptive loop.
  - `confidence_threshold`    -- contact the closest shard, ask the judge;
    expand to the next-closest shard only if confidence is below tau.
  - `progressive_expansion`   -- same expansion loop, but frames the budget
    as an explicit hard stop alongside the confidence stop.
  - `budgeted_communication`  -- no early stop on confidence at all; greedily
    contact shards in centroid-rank order up to a hard cost cap, because the
    guarantee it offers is about worst-case cost, not adaptive efficiency.

Every strategy shares one signature and returns one RoutingTrace, so the
harness's ContextAssembler and evals/run.py never need to know which
strategy actually ran.
"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from lulu_router.cost import Budget, Cost
from lulu_router.shard import Shard, SearchResult
from lulu_router.trace import ExpansionRound, RoutingTrace, ShardScore


class Judge(Protocol):
    """Decides whether accumulated results are sufficient for a query.

    Strategies depend only on this Protocol, never a concrete
    implementation. `judges/geometric.py` (ports local-memory's
    `sigmoid(gap) x coverage`) and `judges/claude_cli.py` (shells out to
    `claude -p --model haiku`) both satisfy it -- see docs/THESIS.md and the
    roadmap for why the backend is meant to be swappable.
    """

    name: str

    def judge(
        self,
        query: str,
        results: list[SearchResult],
        sources_contacted: int,
        total_sources: int,
    ) -> tuple[float, bool, str]:
        """Returns (confidence, should_expand, reasoning)."""
        ...


def _rank_shards_by_centroid(
    shards: list[Shard], query_vec: np.ndarray, scope: str | None
) -> list[tuple[Shard, float]]:
    """Shards the caller's scope doesn't permit are excluded entirely --
    they never even appear in shards_considered. That's deliberate: a
    denied shard isn't a routing decision, it's an access-control boundary,
    and the two shouldn't be visually or statistically conflated in a
    trace."""
    scored = []
    for shard in shards:
        if not shard.permits(scope):
            continue
        sim = float(shard.centroid @ query_vec) if shard.centroid is not None else 0.0
        scored.append((shard, sim))
    return sorted(scored, key=lambda pair: -pair[1])


def query_all(
    query: str,
    query_vec: np.ndarray,
    shards: list[Shard],
    budget: Budget,
    k: int,
    judge: Judge,
    scope: str | None = None,
) -> RoutingTrace:
    considered: list[ShardScore] = []
    spent = Cost()
    all_results: list[SearchResult] = []

    for shard, sim in _rank_shards_by_centroid(shards, query_vec, scope):
        results = shard.search(query_vec, k, query=query)
        spent = spent + shard.cost.cost_for(len(results))
        all_results.extend(results)
        considered.append(ShardScore(shard.id, sim, contacted=True))

    all_results.sort(key=lambda r: -r.score)
    confidence, _, _ = judge.judge(query, all_results[:k], len(considered), len(considered))

    return RoutingTrace(
        query=query,
        strategy="query_all",
        judge=judge.name,
        budget=budget,
        shards_considered=considered,
        confidence=confidence,
        spent=spent,
        results=all_results[:k],
    )


def flat_topk(
    query: str,
    query_vec: np.ndarray,
    shards: list[Shard],
    budget: Budget,
    k: int,
    judge: Judge,
    scope: str | None = None,
) -> RoutingTrace:
    trace = query_all(query, query_vec, shards, budget, k, judge, scope)
    trace.strategy = "flat_topk"
    return trace


def top_n_neighbors(
    query: str,
    query_vec: np.ndarray,
    shards: list[Shard],
    budget: Budget,
    k: int,
    judge: Judge,
    scope: str | None = None,
    n: int = 3,
) -> RoutingTrace:
    ranked = _rank_shards_by_centroid(shards, query_vec, scope)
    considered: list[ShardScore] = []
    spent = Cost()
    all_results: list[SearchResult] = []

    for i, (shard, sim) in enumerate(ranked):
        if i < n:
            results = shard.search(query_vec, k, query=query)
            spent = spent + shard.cost.cost_for(len(results))
            all_results.extend(results)
            considered.append(ShardScore(shard.id, sim, contacted=True))
        else:
            considered.append(ShardScore(shard.id, sim, contacted=False, skip_reason="outside_top_n"))

    all_results.sort(key=lambda r: -r.score)
    confidence, _, _ = judge.judge(
        query, all_results[:k], sum(1 for s in considered if s.contacted), len(ranked)
    )

    return RoutingTrace(
        query=query,
        strategy="top_n_neighbors",
        judge=judge.name,
        budget=budget,
        shards_considered=considered,
        confidence=confidence,
        spent=spent,
        results=all_results[:k],
    )


def confidence_threshold(
    query: str,
    query_vec: np.ndarray,
    shards: list[Shard],
    budget: Budget,
    k: int,
    judge: Judge,
    scope: str | None = None,
    tau: float = 0.6,
) -> RoutingTrace:
    """Contact the single closest shard. If the judge says confidence is
    below tau, expand to the next-closest shard and re-judge. Stops the
    moment confidence clears tau, when the budget runs out, or when shards
    run out."""
    ranked = _rank_shards_by_centroid(shards, query_vec, scope)
    considered: list[ShardScore] = []
    rounds: list[ExpansionRound] = []
    spent = Cost()
    all_results: list[SearchResult] = []
    confidence = 0.0
    stop_reason = "confidence_met"

    for i, (shard, sim) in enumerate(ranked):
        if spent.exceeds(budget):
            stop_reason = "budget_exhausted"
            break

        results = shard.search(query_vec, k, query=query)
        spent = spent + shard.cost.cost_for(len(results))
        all_results.extend(results)
        considered.append(ShardScore(shard.id, sim, contacted=True))
        all_results.sort(key=lambda r: -r.score)

        confidence_before = confidence
        confidence, should_expand, reasoning = judge.judge(
            query, all_results[:k], i + 1, len(ranked)
        )
        verdict = "expand" if (should_expand and confidence < tau) else "sufficient"
        rounds.append(
            ExpansionRound(
                round_index=i,
                shard_id=shard.id,
                confidence_before=confidence_before,
                confidence_after=confidence,
                judge=judge.name,
                verdict=verdict,
                reasoning=reasoning,
            )
        )
        if verdict == "sufficient":
            break
    else:
        stop_reason = "sources_exhausted"

    contacted_ids = {s.shard_id for s in considered}
    for shard, sim in ranked:
        if shard.id not in contacted_ids:
            considered.append(ShardScore(shard.id, sim, contacted=False, skip_reason=stop_reason))

    return RoutingTrace(
        query=query,
        strategy="confidence_threshold",
        judge=judge.name,
        budget=budget,
        shards_considered=considered,
        rounds=rounds,
        confidence=confidence,
        spent=spent,
        results=all_results[:k],
    )


def progressive_expansion(
    query: str,
    query_vec: np.ndarray,
    shards: list[Shard],
    budget: Budget,
    k: int,
    judge: Judge,
    scope: str | None = None,
    tau: float = 0.6,
) -> RoutingTrace:
    """Same expansion loop as confidence_threshold, but the budget is
    checked *before committing* to each round's cost -- a round whose
    projected cost would blow the budget is refused even if the current
    spend hasn't technically exceeded it yet (as long as at least one
    shard has already been contacted, so a query is never left empty)."""
    ranked = _rank_shards_by_centroid(shards, query_vec, scope)
    considered: list[ShardScore] = []
    rounds: list[ExpansionRound] = []
    spent = Cost()
    all_results: list[SearchResult] = []
    confidence = 0.0
    stop_reason = "confidence_met"

    for i, (shard, sim) in enumerate(ranked):
        if spent.exceeds(budget):
            stop_reason = "budget_exhausted"
            break

        results = shard.search(query_vec, k, query=query)
        projected = spent + shard.cost.cost_for(len(results))
        if projected.exceeds(budget) and i > 0:
            stop_reason = "budget_exhausted"
            break

        spent = projected
        all_results.extend(results)
        considered.append(ShardScore(shard.id, sim, contacted=True))
        all_results.sort(key=lambda r: -r.score)

        confidence_before = confidence
        confidence, should_expand, reasoning = judge.judge(
            query, all_results[:k], i + 1, len(ranked)
        )
        verdict = "sufficient" if (confidence >= tau or not should_expand) else "expand"
        rounds.append(
            ExpansionRound(
                round_index=i,
                shard_id=shard.id,
                confidence_before=confidence_before,
                confidence_after=confidence,
                judge=judge.name,
                verdict=verdict,
                reasoning=reasoning,
            )
        )
        if verdict == "sufficient":
            break
    else:
        stop_reason = "sources_exhausted"

    contacted_ids = {s.shard_id for s in considered}
    for shard, sim in ranked:
        if shard.id not in contacted_ids:
            considered.append(ShardScore(shard.id, sim, contacted=False, skip_reason=stop_reason))

    return RoutingTrace(
        query=query,
        strategy="progressive_expansion",
        judge=judge.name,
        budget=budget,
        shards_considered=considered,
        rounds=rounds,
        confidence=confidence,
        spent=spent,
        results=all_results[:k],
    )


def budgeted_communication(
    query: str,
    query_vec: np.ndarray,
    shards: list[Shard],
    budget: Budget,
    k: int,
    judge: Judge,
    scope: str | None = None,
) -> RoutingTrace:
    """No early stop on confidence. Greedily contact shards in
    centroid-rank order, using a worst-case cost estimate (cost_for(k),
    before the search even runs) to decide whether the *next* shard fits
    under the cap. This is the strategy to reach for when the requirement
    is a hard ceiling on cost, not adaptive efficiency."""
    ranked = _rank_shards_by_centroid(shards, query_vec, scope)
    considered: list[ShardScore] = []
    spent = Cost()
    all_results: list[SearchResult] = []

    for shard, sim in ranked:
        worst_case = shard.cost.cost_for(k)
        if (spent + worst_case).exceeds(budget):
            considered.append(ShardScore(shard.id, sim, contacted=False, skip_reason="budget_exhausted"))
            continue
        results = shard.search(query_vec, k, query=query)
        spent = spent + shard.cost.cost_for(len(results))
        all_results.extend(results)
        considered.append(ShardScore(shard.id, sim, contacted=True))

    all_results.sort(key=lambda r: -r.score)
    confidence, _, _ = judge.judge(
        query, all_results[:k], sum(1 for s in considered if s.contacted), len(ranked)
    )

    return RoutingTrace(
        query=query,
        strategy="budgeted_communication",
        judge=judge.name,
        budget=budget,
        shards_considered=considered,
        confidence=confidence,
        spent=spent,
        results=all_results[:k],
    )


STRATEGIES: dict[str, Callable[..., RoutingTrace]] = {
    "query_all": query_all,
    "flat_topk": flat_topk,
    "top_n_neighbors": top_n_neighbors,
    "confidence_threshold": confidence_threshold,
    "progressive_expansion": progressive_expansion,
    "budgeted_communication": budgeted_communication,
}
