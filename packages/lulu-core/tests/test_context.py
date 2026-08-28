"""ContextAssembler: the integration seam. Uses lulu_router's real
InMemoryShardStore-backed shards and a real GeometricJudge -- this is the
one place in lulu-core allowed to know lulu_router exists, so it's the
one place that should exercise the real thing rather than a fake."""

from __future__ import annotations

import numpy as np
import pytest
from lulu_router.backends.memory import InMemoryShardStore
from lulu_router.cost import Budget, CostProfile
from lulu_router.judges.geometric import GeometricJudge
from lulu_router.router import MemoryRouter
from lulu_router.shard import Shard

from lulu.context import ContextAssembler


def _shard(shard_id: str, contents: list[str], vectors: np.ndarray) -> Shard:
    store = InMemoryShardStore.from_vectors(
        ids=[f"{shard_id}-{i}" for i in range(len(contents))],
        contents=contents,
        vectors=vectors,
    )
    centroid = vectors.mean(axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    return Shard(
        id=shard_id,
        store=store,
        cost=CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=50),
        centroid=centroid,
    )


@pytest.fixture
def router() -> MemoryRouter:
    rng = np.random.default_rng(0)
    shard_a = _shard("episodic", ["decided to use SQLite", "session started at 9am"], rng.normal(size=(2, 8)))
    shard_b = _shard("code", ["def foo(): pass", "class Bar: pass"], rng.normal(size=(2, 8)) + 10)
    return MemoryRouter(shards=[shard_a, shard_b], judge=GeometricJudge())


def test_assemble_returns_rendered_text_and_trace(router: MemoryRouter):
    assembler = ContextAssembler(router=router, strategy="query_all", k=5)
    query_vec = router.shards[0].centroid  # dead-center on shard_a

    result = assembler.assemble("what did we decide", query_vec, Budget())

    assert "# Relevant memory" in result.text
    assert result.trace.strategy == "query_all"
    assert len(result.trace.results) > 0


def test_assemble_empty_results_returns_empty_text():
    router = MemoryRouter(shards=[], judge=GeometricJudge())
    assembler = ContextAssembler(router=router)

    result = assembler.assemble("anything", np.zeros(8), Budget())

    assert result.text == ""
    assert result.trace.results == []


def test_assemble_respects_scope_restriction(router: MemoryRouter):
    router.shards[1].allowed_scopes = frozenset({"customer-a"})
    assembler = ContextAssembler(router=router, strategy="query_all", k=5)
    query_vec = router.shards[1].centroid

    result = assembler.assemble("code question", query_vec, Budget(), scope="customer-b")

    contacted_shard_ids = {s.shard_id for s in result.trace.shards_considered}
    assert "code" not in contacted_shard_ids
