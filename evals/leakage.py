"""evals/leakage.py: the permission-boundary demo. Binary, no
interpretation needed -- see docs/THESIS.md, Contribution #1.

The comparison has to be built carefully to actually test the claim.
lulu_router's own `flat_topk` strategy still respects `Shard.permits()`,
because permission filtering happens at shard-ranking time, BEFORE any
strategy runs -- every strategy, flat_topk included, already can't see a
shard the caller's scope doesn't permit. That's correct behavior for the
router, but it means re-running flat_topk with a scope argument does NOT
demonstrate the actual claim, which is that a genuinely flat index --
one where per-customer vectors are merged into a single pool with no
shard boundary left to attach a permission check to -- cannot express
the boundary at all, not just chooses not to.

So this script builds a *real* flat index (merge every shard's vectors
into one InMemoryShardStore, discard which shard/scope each one came
from) and searches it directly, with no scope parameter possible, because
there is nothing left to check a scope against. That is the actual
mechanism gap the thesis rests on, made concrete instead of asserted.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from lulu_router.backends.memory import InMemoryShardStore
from lulu_router.cost import Budget, CostProfile
from lulu_router.judges.geometric import GeometricJudge
from lulu_router.router import MemoryRouter
from lulu_router.shard import Shard

RNG_SEED = 42
DIM = 16
MEMORIES_PER_CUSTOMER = 30
N_QUERIES = 200
K = 5
# Distance between the two customers' cluster centers, relative to
# within-cluster spread (scale=0.4 per point, see _build_customer_memories).
# Small values make the clusters genuinely overlap in vector space -- the
# realistic case (e.g. two customers both writing about "invoice" or
# "login issue"), and the case that actually stresses the claim: if the
# flat index doesn't leak when the clusters are this close, the test
# proves nothing. Empirically swept (not guessed): 0.5 reliably produces
# real, non-trivial flat-index leakage (~25% of queries) while Lulu stays
# at 0 regardless -- separation >= 1.5 stops leaking for either at
# n=200/k=5, which just means the topics are too distinct to be a
# meaningful test, not that the flat index is actually safe.
SEPARATION = 0.5


@dataclass
class LeakageResult:
    label: str
    queries: int
    leaked_queries: int
    leaked_memories_total: int

    @property
    def leak_rate(self) -> float:
        return self.leaked_queries / self.queries if self.queries else 0.0


def _build_customer_memories(rng: np.random.Generator, customer: str, center: np.ndarray) -> tuple[list[str], list[str], np.ndarray]:
    ids, contents = [], []
    vectors = center + rng.normal(scale=0.4, size=(MEMORIES_PER_CUSTOMER, DIM))
    for i in range(MEMORIES_PER_CUSTOMER):
        ids.append(f"{customer}-{i}")
        contents.append(f"{customer} confidential record {i}")
    return ids, contents, vectors.astype(np.float32)


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def build_scenario(rng: np.random.Generator, separation: float = SEPARATION):
    center_a = rng.normal(size=DIM) * 5
    center_b = center_a + rng.normal(size=DIM) * separation

    ids_a, contents_a, vecs_a = _build_customer_memories(rng, "customer-a", center_a)
    ids_b, contents_b, vecs_b = _build_customer_memories(rng, "customer-b", center_b)

    cost = CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=80)

    shard_a = Shard(
        id="customer-a-memory",
        store=InMemoryShardStore.from_vectors(ids_a, contents_a, vecs_a),
        cost=cost,
        allowed_scopes=frozenset({"customer-a"}),
    )
    shard_a.centroid = _normalize_rows(vecs_a.mean(axis=0, keepdims=True))[0]

    shard_b = Shard(
        id="customer-b-memory",
        store=InMemoryShardStore.from_vectors(ids_b, contents_b, vecs_b),
        cost=cost,
        allowed_scopes=frozenset({"customer-b"}),
    )
    shard_b.centroid = _normalize_rows(vecs_b.mean(axis=0, keepdims=True))[0]

    # The genuinely flat index: every vector, merged, with no record of
    # which customer/scope it came from -- there is nothing left for a
    # permission check to attach to.
    flat_index = InMemoryShardStore.from_vectors(
        ids_a + ids_b, contents_a + contents_b, np.vstack([vecs_a, vecs_b])
    )

    router = MemoryRouter(shards=[shard_a, shard_b], judge=GeometricJudge())
    return router, flat_index, ids_a, ids_b, _normalize_rows(vecs_a)


def is_customer_b_id(memory_id: str) -> bool:
    return memory_id.startswith("customer-b-")


def run(
    n_queries: int = N_QUERIES, k: int = K, strategy: str = "query_all", separation: float = SEPARATION
) -> tuple[LeakageResult, LeakageResult]:
    rng = np.random.default_rng(RNG_SEED)
    router, flat_index, ids_a, _ids_b, query_vecs_a = build_scenario(rng, separation=separation)
    budget = Budget(max_tokens=100_000, max_latency_ms=60_000.0, max_usd=10.0)

    # size=n_queries with replace=True: genuinely draw n_queries queries,
    # reusing the (small, fixed) pool of customer-a vectors as needed --
    # min()-capping this to the pool size (an earlier bug in this script)
    # silently ran fewer queries than requested every time.
    query_indices = rng.choice(len(query_vecs_a), size=n_queries, replace=True)

    lulu_leaked_queries = 0
    lulu_leaked_total = 0
    flat_leaked_queries = 0
    flat_leaked_total = 0

    for idx in query_indices:
        query_vec = query_vecs_a[idx]

        # Lulu: routed, scoped to customer-a.
        trace = router.route(f"query-{idx}", query_vec, strategy=strategy, budget=budget, k=k, scope="customer-a")
        leaked = sum(1 for r in trace.results if is_customer_b_id(r.id))
        if leaked > 0:
            lulu_leaked_queries += 1
            lulu_leaked_total += leaked

        # Flat index: no scope concept at all -- searched directly.
        flat_results = flat_index.search(query_vec, k)
        leaked_flat = sum(1 for r in flat_results if is_customer_b_id(r.id))
        if leaked_flat > 0:
            flat_leaked_queries += 1
            flat_leaked_total += leaked_flat

    n = len(query_indices)
    return (
        LeakageResult("lulu (scoped routing)", n, lulu_leaked_queries, lulu_leaked_total),
        LeakageResult("flat index (no shard boundary)", n, flat_leaked_queries, flat_leaked_total),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-queries", type=int, default=N_QUERIES)
    parser.add_argument("--k", type=int, default=K)
    parser.add_argument(
        "--separation",
        type=float,
        default=SEPARATION,
        help="distance between the two customers' cluster centers -- lower means more "
        "realistic (topically overlapping) memories and a harder test",
    )
    parser.add_argument(
        "--strategy",
        default="query_all",
        choices=["query_all", "flat_topk", "top_n_neighbors", "confidence_threshold", "progressive_expansion", "budgeted_communication"],
        help="Lulu's routing strategy under test. Every one of them is scope-checked at the "
        "same point (before strategy dispatch), so the choice here shouldn't change the "
        "result -- run with a few different ones to confirm that's actually true.",
    )
    args = parser.parse_args()

    lulu_result, flat_result = run(
        n_queries=args.n_queries, k=args.k, strategy=args.strategy, separation=args.separation
    )

    print(
        f"=== Leakage test: {args.n_queries} queries from a customer-a-scoped agent, "
        f"k={args.k}, strategy={args.strategy}, separation={args.separation} ===\n"
    )
    for result in (lulu_result, flat_result):
        print(
            f"{result.label:32s} leaked in {result.leaked_queries}/{result.queries} queries "
            f"({result.leak_rate * 100:.1f}%), {result.leaked_memories_total} customer-b memories total"
        )

    print()
    if lulu_result.leaked_queries == 0 and flat_result.leaked_queries > 0:
        print("PASS: Lulu's scoped routing leaked 0 queries. The flat index leaked because it has no boundary to enforce.")
    elif lulu_result.leaked_queries == 0 and flat_result.leaked_queries == 0:
        print(
            "INCONCLUSIVE: neither leaked -- the flat index never happened to retrieve a "
            "customer-b memory in these queries. Try raising CLUSTER_OVERLAP or --n-queries."
        )
    else:
        print(f"FAIL: Lulu's scoped routing leaked in {lulu_result.leaked_queries} queries. This should never happen -- investigate.")


if __name__ == "__main__":
    main()
