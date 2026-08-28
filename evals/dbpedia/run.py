"""Reproduces the distributed-vector-memory-routing paper's DBpedia14
experiment on top of lulu_router's own strategies/judges, and adds the
axis the paper didn't have: geometric confidence vs. an LLM-judge stopping
criterion.

Methodology follows the original notebook
(Research_hub/Distributed Memory Routing/GitHub_Repo/notebook/
dbpedia14_distributed_memory.ipynb) with two deliberate deviations,
documented rather than hidden:

  1. Corpus size defaults small (a few thousand docs, not 100K) so this
     runs in minutes on a dev machine instead of requiring a Kaggle-class
     box. Pass --n-docs 100000 --partition-counts 2,4,8,16 to reproduce
     the original paper's regime.
  2. Embeddings come from FastEmbed (BAAI/bge-small-en-v1.5, local ONNX),
     not sentence-transformers/all-MiniLM-L6-v2. This is the embedding
     stack the harness actually uses (see local-memory/core/embeddings.py)
     -- there's no reason to pull in a second, torch-backed embedding
     pipeline that nothing else in this project touches.

Ground truth is brute-force cosine similarity over the sampled corpus
(same as the paper: within-sample, not against the full 630K DBpedia14).

Usage:
    uv run python evals/dbpedia/run.py
    uv run python evals/dbpedia/run.py --n-docs 100000 --partition-counts 2,4,8,16
    uv run python evals/dbpedia/run.py --llm-judge --llm-judge-queries 5

Caveat on --llm-judge: DBpedia14's "queries" are just other documents used as
similarity probes, not natural-language task questions -- there's no real
task for a sufficiency judge to reason about the way there would be for a
harness turn ("do I have enough context for this coding task?"). Treat the
--llm-judge numbers here as a plumbing check (does ClaudeCLIJudge correctly
wire into every strategy and produce sane output), not as the real
geometric-vs-LLM-judge quality comparison -- that comparison belongs in
evals/agent_tasks (day 6), where queries are actual task descriptions.
"""

from __future__ import annotations

import argparse
import itertools
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from lulu_router.cost import Budget, CostProfile
from lulu_router.judges.claude_cli import ClaudeCLIJudge
from lulu_router.judges.geometric import GeometricJudge
from lulu_router.partition import kmeans_partition
from lulu_router.strategies import STRATEGIES

CACHE_DIR = Path(__file__).parent / ".cache"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384

# Uniform placeholder cost: this eval measures communication (shards
# contacted) and retrieval quality (recall/MRR/NDCG), exactly like the
# paper -- heterogeneous $/latency cost between shard *types* is what
# evals/agent_tasks (day 6) measures, once the harness's real shards
# (local vs. MCP-connector) exist.
UNIFORM_COST = CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=1)
GENEROUS_BUDGET = Budget(max_tokens=10_000, max_latency_ms=60_000.0, max_usd=1.0)


def budget_for(strategy_name: str, n_partitions: int, k: int) -> Budget:
    """Every strategy except budgeted_communication should be free to
    spend as needed -- their whole point is deciding *when* enough is
    enough, not living under a cap. budgeted_communication's entire
    identity is the cap, so it needs one that can actually bind; a
    max_contacted = K // 2 hard limit is exactly what the paper's own
    budgeted strategy used. Without this, a "generous" budget never binds
    and the strategy silently degenerates into query_all."""
    if strategy_name == "budgeted_communication":
        max_shards = max(2, n_partitions // 2)
        return Budget(
            max_tokens=max_shards * k * UNIFORM_COST.tokens_per_result,
            max_latency_ms=60_000.0,
            max_usd=1.0,
        )
    return GENEROUS_BUDGET

STRATEGY_LABELS = {
    "query_all": "Global (baseline)",
    "flat_topk": "Flat top-k (skeptic's baseline)",
    "top_n_neighbors": "Top-N Neighbors",
    "confidence_threshold": "Confidence Threshold",
    "progressive_expansion": "Progressive Expansion",
    "budgeted_communication": "Budgeted Communication",
}

# Strategies whose routing decision actually depends on the judge's
# should_expand verdict. The others (query_all, flat_topk,
# top_n_neighbors, budgeted_communication) call judge.judge() once just to
# report a confidence number on the trace -- swapping their judge changes
# nothing about which shards get contacted, so there's no point paying for
# an LLM call on those when comparing judges.
JUDGE_SENSITIVE_STRATEGIES = ("confidence_threshold", "progressive_expansion")


def load_corpus(n_docs: int, seed: int) -> tuple[list[str], list[int]]:
    """Streams the first n_docs examples of DBpedia14's train split --
    streaming avoids materializing the full ~560K-row split just to take a
    small slice."""
    from datasets import load_dataset

    ds = load_dataset("fancyzhx/dbpedia_14", split="train", streaming=True)
    examples = list(itertools.islice(ds, n_docs))
    texts = [ex["content"] for ex in examples]
    labels = [ex["label"] for ex in examples]
    return texts, labels


def embed_corpus(texts: list[str], cache_key: str) -> np.ndarray:
    cache_path = CACHE_DIR / f"{cache_key}.npy"
    if cache_path.exists():
        print(f"[embed] using cached embeddings: {cache_path}")
        return np.load(cache_path)

    from fastembed import TextEmbedding

    print(f"[embed] embedding {len(texts):,} docs with {EMBEDDING_MODEL} ...")
    t0 = time.time()
    model = TextEmbedding(EMBEDDING_MODEL)
    vectors = np.array(list(model.embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms
    print(f"[embed] done in {time.time() - t0:.1f}s")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    return vectors


def ground_truth(embeddings: np.ndarray, query_idx: int, k: int) -> set[int]:
    query_vec = embeddings[query_idx]
    sims = embeddings @ query_vec
    top = np.argsort(-sims)[:k]
    return set(int(i) for i in top)


def compute_recall(gt: set[int], predicted: list[int]) -> float:
    if not gt:
        return 0.0
    hits = sum(1 for p in predicted if p in gt)
    return hits / len(gt)


def compute_mrr(gt: set[int], predicted: list[int]) -> float:
    for i, p in enumerate(predicted):
        if p in gt:
            return 1.0 / (i + 1)
    return 0.0


def compute_ndcg(gt: set[int], predicted: list[int], k: int) -> float:
    relevance = np.array([1.0 if p in gt else 0.0 for p in predicted[:k]])
    if relevance.sum() == 0:
        return 0.0
    positions = np.arange(1, len(relevance) + 1)
    dcg = float(np.sum(relevance / np.log2(positions + 1)))
    ideal = np.sort(relevance)[::-1]
    idcg = float(np.sum(ideal / np.log2(positions + 1)))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class StrategyRun:
    strategy: str
    judge: str
    n_partitions: int
    recall: float
    mrr: float
    ndcg: float
    shards_contacted: float
    comm_reduction_pct: float


def run_sweep(
    embeddings: np.ndarray,
    texts: list[str],
    partition_counts: list[int],
    n_queries: int,
    k: int,
    seed: int,
    llm_judge: bool,
    llm_judge_queries: int,
) -> list[StrategyRun]:
    n_docs = len(embeddings)
    rng = np.random.default_rng(seed)
    query_indices = rng.choice(n_docs, size=min(n_queries, n_docs), replace=False)
    ids = [str(i) for i in range(n_docs)]

    geometric = GeometricJudge()
    claude_cli = ClaudeCLIJudge() if llm_judge else None

    rows: list[StrategyRun] = []

    for n_partitions in partition_counts:
        print(f"\n=== K={n_partitions} partitions ===")
        # Real document text, not a placeholder -- the LLM judge has to
        # read actual content to judge sufficiency. An earlier version of
        # this script passed index strings as "content" and a fake
        # "doc-<id>" string as the query; the LLM judge, given nothing
        # real to reason about, collapsed to near-instant (and wrong)
        # "sufficient" verdicts. That failure was in this eval script, not
        # in ClaudeCLIJudge itself -- worth noting since it's exactly the
        # kind of silent-garbage-in problem evals/dbpedia exists to catch.
        shards = kmeans_partition(
            ids=ids,
            contents=texts,
            vectors=embeddings,
            n_shards=n_partitions,
            cost=UNIFORM_COST,
            random_state=seed,
        )

        for strategy_name, strategy_fn in STRATEGIES.items():
            judges_to_run: list[tuple[str, object, int]] = [("geometric", geometric, len(query_indices))]
            if llm_judge and strategy_name in JUDGE_SENSITIVE_STRATEGIES:
                judges_to_run.append(("claude_cli", claude_cli, min(llm_judge_queries, len(query_indices))))

            for judge_name, judge, n_q in judges_to_run:
                recalls, mrrs, ndcgs, contacted = [], [], [], []
                budget = budget_for(strategy_name, n_partitions, k)
                for qi in query_indices[:n_q]:
                    query_vec = embeddings[qi]
                    gt = ground_truth(embeddings, qi, k)
                    trace = strategy_fn(
                        query=texts[qi][:200],
                        query_vec=query_vec,
                        shards=shards,
                        budget=budget,
                        k=k,
                        judge=judge,
                    )
                    predicted = [int(r.id) for r in trace.results]
                    recalls.append(compute_recall(gt, predicted))
                    mrrs.append(compute_mrr(gt, predicted))
                    ndcgs.append(compute_ndcg(gt, predicted, k))
                    contacted.append(len(trace.shards_contacted))

                avg_contacted = float(np.mean(contacted))
                row = StrategyRun(
                    strategy=strategy_name,
                    judge=judge_name,
                    n_partitions=n_partitions,
                    recall=float(np.mean(recalls)),
                    mrr=float(np.mean(mrrs)),
                    ndcg=float(np.mean(ndcgs)),
                    shards_contacted=avg_contacted,
                    comm_reduction_pct=(1 - avg_contacted / n_partitions) * 100,
                )
                rows.append(row)
                label = STRATEGY_LABELS[strategy_name]
                print(
                    f"  {label:32s} [{judge_name:9s}] "
                    f"Recall@{k}={row.recall:.4f}  MRR={row.mrr:.4f}  NDCG={row.ndcg:.4f}  "
                    f"Comm={row.shards_contacted:.1f}/{n_partitions} ({row.comm_reduction_pct:.0f}% saved)  "
                    f"(n={n_q} queries)"
                )

    return rows


def print_headline(rows: list[StrategyRun], partition_counts: list[int]) -> None:
    print("\n" + "=" * 78)
    print("HEADLINE: does routing beat the flat-index skeptic's baseline?")
    print("=" * 78)
    for n in partition_counts:
        flat = next(r for r in rows if r.strategy == "flat_topk" and r.n_partitions == n and r.judge == "geometric")
        best_routed = max(
            (r for r in rows if r.judge == "geometric" and r.n_partitions == n and r.strategy not in ("query_all", "flat_topk")),
            key=lambda r: r.recall - 0.01 * r.shards_contacted,  # prefer high recall, low comm, roughly
        )
        verdict = "ROUTING WINS" if best_routed.recall >= flat.recall - 1e-6 and best_routed.shards_contacted < flat.shards_contacted else "flat_topk not beaten"
        print(
            f"K={n:2d} | flat_topk: recall={flat.recall:.4f} comm={flat.shards_contacted:.1f}/{n}  "
            f"vs  best routed ({best_routed.strategy}): recall={best_routed.recall:.4f} "
            f"comm={best_routed.shards_contacted:.1f}/{n}  -> {verdict}"
        )

    geo_vs_llm = [r for r in rows if r.judge == "claude_cli"]
    if geo_vs_llm:
        print("\n--- geometric judge vs. LLM judge (on judge-sensitive strategies) ---")
        for llm_row in geo_vs_llm:
            geo_row = next(
                r
                for r in rows
                if r.judge == "geometric"
                and r.strategy == llm_row.strategy
                and r.n_partitions == llm_row.n_partitions
            )
            print(
                f"K={llm_row.n_partitions:2d} {llm_row.strategy:24s} "
                f"geometric: recall={geo_row.recall:.4f} comm={geo_row.shards_contacted:.1f}  |  "
                f"llm: recall={llm_row.recall:.4f} comm={llm_row.shards_contacted:.1f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-docs", type=int, default=3000)
    parser.add_argument("--partition-counts", type=str, default="2,4,8")
    parser.add_argument("--n-queries", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--llm-judge", action="store_true", help="also run the judge-sensitive strategies under ClaudeCLIJudge")
    parser.add_argument("--llm-judge-queries", type=int, default=5, help="how many queries to spend on the LLM judge (it's slow and shells out per judgment)")
    args = parser.parse_args()

    partition_counts = [int(x) for x in args.partition_counts.split(",")]

    np.random.seed(args.seed)
    texts, _labels = load_corpus(args.n_docs, args.seed)
    print(f"[corpus] {len(texts):,} DBpedia14 documents loaded")

    embeddings = embed_corpus(texts, cache_key=f"dbpedia14_n{args.n_docs}_{EMBEDDING_MODEL.replace('/', '_')}")

    rows = run_sweep(
        embeddings,
        texts,
        partition_counts,
        n_queries=args.n_queries,
        k=args.top_k,
        seed=args.seed,
        llm_judge=args.llm_judge,
        llm_judge_queries=args.llm_judge_queries,
    )

    print_headline(rows, partition_counts)


if __name__ == "__main__":
    main()
