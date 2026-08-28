"""Shared fixtures: a small synthetic corpus split into three semantically
separated clusters (so centroid-based ranking has something real to rank),
plus a stub Judge so strategies.py can be tested in isolation before
judges/geometric.py and judges/claude_cli.py (day 2) exist.
"""

from __future__ import annotations

import numpy as np
import pytest

from lulu_router.cost import Budget, CostProfile
from lulu_router.partition import kmeans_partition


class StubJudge:
    """Minimal judge for testing strategies.py in isolation. Confidence is
    just the top result's similarity score; expands whenever confidence is
    below 0.9 and more sources remain. Not a stand-in for either real
    judge -- geometric.py and claude_cli.py have their own tests."""

    name = "stub"

    def judge(self, query, results, sources_contacted, total_sources):
        if not results:
            return 0.0, sources_contacted < total_sources, "no results yet"
        confidence = max(0.0, min(1.0, float(results[0].score)))
        should_expand = confidence < 0.9 and sources_contacted < total_sources
        return confidence, should_expand, f"top_score={confidence:.2f}"


@pytest.fixture
def judge():
    return StubJudge()


@pytest.fixture
def synthetic_corpus():
    """Three tight clusters in 8-d space, well separated, so centroid
    ranking and top-k are unambiguous to assert on. Returns
    (ids, contents, vectors, metadata, cluster_centers)."""
    rng = np.random.default_rng(0)
    dim = 8
    centers = rng.normal(size=(3, dim)) * 5
    ids: list[str] = []
    contents: list[str] = []
    vectors: list[np.ndarray] = []
    metadata: list[dict] = []
    for cluster_id, center in enumerate(centers):
        for i in range(20):
            vec = center + rng.normal(scale=0.3, size=dim)
            ids.append(f"c{cluster_id}-{i}")
            contents.append(f"memory {cluster_id}-{i}")
            vectors.append(vec)
            metadata.append({"cluster": cluster_id})
    return ids, contents, np.array(vectors, dtype=np.float32), metadata, centers


@pytest.fixture
def cheap_cost():
    return CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=80)


@pytest.fixture
def expensive_cost():
    return CostProfile(latency_ms=800.0, usd_per_query=0.002, tokens_per_result=300)


@pytest.fixture
def shards(synthetic_corpus, cheap_cost):
    ids, contents, vectors, metadata, _ = synthetic_corpus
    return kmeans_partition(ids, contents, vectors, n_shards=3, cost=cheap_cost)


@pytest.fixture
def budget():
    return Budget(max_tokens=5000, max_latency_ms=5000.0, max_usd=1.0)


def query_vec_for_cluster(synthetic_corpus, cluster_id: int) -> np.ndarray:
    """A query embedding dead-center in one synthetic cluster, normalized
    to unit length like every stored vector."""
    _, _, _, _, centers = synthetic_corpus
    v = centers[cluster_id]
    return v / np.linalg.norm(v)
