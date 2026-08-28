"""Semantic sharding: partition a corpus into K shards via KMeans on
embeddings, one shard per cluster, each carrying its cluster centroid. This
is the paper's partitioning strategy (distributed-vector-memory-routing),
lifted from a single DBpedia14 index into a general corpus -> shards step.

Cost and access-scope are deliberately NOT derived here: they come from
where a shard's storage actually lives (local disk vs. an MCP connector) or
from who is allowed to read it, neither of which follows from the semantic
content of the shard. The harness's ContextAssembler assigns those after
partitioning.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

from lulu_router.backends.memory import InMemoryShardStore
from lulu_router.cost import CostProfile
from lulu_router.shard import Shard


def kmeans_partition(
    ids: list[str],
    contents: list[str],
    vectors: np.ndarray,
    n_shards: int,
    cost: CostProfile,
    shard_prefix: str = "shard",
    metadata: list[dict] | None = None,
    random_state: int = 0,
) -> list[Shard]:
    """Cluster `vectors` into `n_shards` groups and return one Shard per
    non-empty cluster, each carrying its cluster centroid and the given
    CostProfile.
    """
    vectors = np.asarray(vectors, dtype=np.float32)
    if len(ids) == 0:
        return []
    n_shards = max(1, min(n_shards, len(ids)))
    metadata = metadata or [{} for _ in ids]

    km = KMeans(n_clusters=n_shards, n_init="auto", random_state=random_state)
    labels = km.fit_predict(vectors)

    shards: list[Shard] = []
    for cluster_id in range(n_shards):
        member_idx = np.where(labels == cluster_id)[0]
        if len(member_idx) == 0:
            continue
        cluster_vectors = vectors[member_idx]
        centroid = cluster_vectors.mean(axis=0)
        centroid = centroid / max(float(np.linalg.norm(centroid)), 1e-9)

        store = InMemoryShardStore.from_vectors(
            ids=[ids[i] for i in member_idx],
            contents=[contents[i] for i in member_idx],
            vectors=cluster_vectors,
            metadata=[metadata[i] for i in member_idx],
        )
        shards.append(
            Shard(
                id=f"{shard_prefix}-{cluster_id}",
                store=store,
                cost=cost,
                centroid=centroid,
            )
        )
    return shards
