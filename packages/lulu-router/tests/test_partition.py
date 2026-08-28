"""KMeans partitioning: each shard should get a real centroid, and points
should land in the shard whose semantic cluster they actually belong to."""

from __future__ import annotations

import numpy as np

from lulu_router.partition import kmeans_partition


def test_partition_produces_requested_shard_count(synthetic_corpus, cheap_cost):
    ids, contents, vectors, metadata, _ = synthetic_corpus
    shards = kmeans_partition(ids, contents, vectors, n_shards=3, cost=cheap_cost)
    assert len(shards) == 3
    assert sum(len(s) for s in shards) == len(ids)


def test_shard_centroids_are_normalized(shards):
    for shard in shards:
        assert shard.centroid is not None
        assert np.isclose(np.linalg.norm(shard.centroid), 1.0, atol=1e-5)


def test_membership_matches_synthetic_cluster_labels(synthetic_corpus, cheap_cost):
    """The synthetic clusters are separated enough (centers scaled by 5,
    within-cluster noise scale 0.3) that KMeans shouldn't mix them: every
    point in a resulting shard should share the same ground-truth label."""
    ids, contents, vectors, metadata, _ = synthetic_corpus
    shards = kmeans_partition(ids, contents, vectors, n_shards=3, cost=cheap_cost)
    id_to_cluster = {i: m["cluster"] for i, m in zip(ids, metadata)}
    for shard in shards:
        clusters_in_shard = {id_to_cluster[i] for i in shard.store.ids}
        assert len(clusters_in_shard) == 1


def test_each_shard_carries_the_given_cost_profile(synthetic_corpus, cheap_cost):
    ids, contents, vectors, metadata, _ = synthetic_corpus
    shards = kmeans_partition(ids, contents, vectors, n_shards=3, cost=cheap_cost)
    for shard in shards:
        assert shard.cost is cheap_cost
