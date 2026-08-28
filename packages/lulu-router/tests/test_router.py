"""MemoryRouter: the dispatch surface the harness actually calls. Thin by
design -- the behavior under test lives in test_strategies.py."""

from __future__ import annotations

import pytest

from lulu_router.router import MemoryRouter
from .conftest import query_vec_for_cluster


def test_route_dispatches_to_the_named_strategy(shards, judge, budget, synthetic_corpus):
    router = MemoryRouter(shards=shards, judge=judge)
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    trace = router.route("q", qvec, strategy="top_n_neighbors", budget=budget, k=5)
    assert trace.strategy == "top_n_neighbors"


def test_route_uses_default_strategy_when_unspecified(shards, judge, budget, synthetic_corpus):
    router = MemoryRouter(shards=shards, judge=judge)
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    trace = router.route("q", qvec, budget=budget, k=5)
    assert trace.strategy == "progressive_expansion"


def test_route_rejects_unknown_strategy(shards, judge, synthetic_corpus):
    router = MemoryRouter(shards=shards, judge=judge)
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    with pytest.raises(ValueError, match="unknown strategy"):
        router.route("q", qvec, strategy="not_a_real_strategy")


def test_route_without_budget_uses_default_budget(shards, judge, synthetic_corpus):
    router = MemoryRouter(shards=shards, judge=judge)
    qvec = query_vec_for_cluster(synthetic_corpus, 0)
    trace = router.route("q", qvec, strategy="query_all")
    assert trace.budget.max_tokens == 2000
