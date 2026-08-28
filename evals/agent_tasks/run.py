"""evals/agent_tasks/run.py: the geometric-vs-LLM-judge comparison
evals/dbpedia can't actually provide, because DBpedia14's "queries" are
just other documents used as similarity probes, not real task questions
(see evals/dbpedia/run.py's docstring). These 25 hand-labeled tasks
(tasks.py) against a small, hand-built project memory bank (memories.py)
are real natural-language questions -- recall here means "did the router
surface the memory a human actually wanted," not "did it find the
nearest vector."

Shards are partitioned by type (semantic/procedural/episodic), matching
memory.py's own design -- not KMeans, which is what a 100K-document
research corpus needs, not a ~18-memory project memory bank.

Usage:
    uv run python evals/agent_tasks/run.py
    uv run python evals/agent_tasks/run.py --llm-judge --llm-judge-tasks 8
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from lulu_router.backends.memory import InMemoryShardStore
from lulu_router.cost import Budget, CostProfile
from lulu_router.judges.claude_cli import ClaudeCLIJudge
from lulu_router.judges.geometric import GeometricJudge
from lulu_router.shard import Shard
from lulu_router.strategies import STRATEGIES

from memories import MEMORIES
from tasks import TASKS

CACHE_DIR = Path(__file__).parent / ".cache"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 3

SHARD_COSTS = {
    "episodic": CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=80),
    "semantic": CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=120),
    "procedural": CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=150),
}

GENEROUS_BUDGET = Budget(max_tokens=100_000, max_latency_ms=60_000.0, max_usd=10.0)

# Same reasoning as evals/dbpedia: only strategies whose routing decision
# actually depends on the judge's verdict are worth re-running under the
# (slow, shells-out-per-call) LLM judge.
JUDGE_SENSITIVE_STRATEGIES = ("confidence_threshold", "progressive_expansion")


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def embed_all(texts: list[str], cache_key: str) -> np.ndarray:
    cache_path = CACHE_DIR / f"{cache_key}.npy"
    if cache_path.exists():
        return np.load(cache_path)

    from fastembed import TextEmbedding

    model = TextEmbedding(EMBEDDING_MODEL)
    vectors = _normalize(np.array(list(model.embed(texts)), dtype=np.float32))

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    return vectors


def build_shards(memory_vectors: dict[str, np.ndarray]) -> list[Shard]:
    by_shard: dict[str, list[tuple[str, str]]] = {"semantic": [], "procedural": [], "episodic": []}
    for m in MEMORIES:
        by_shard[m.shard].append((m.id, m.content))

    shards = []
    for shard_name, entries in by_shard.items():
        ids = [e[0] for e in entries]
        contents = [e[1] for e in entries]
        vectors = np.array([memory_vectors[i] for i in ids], dtype=np.float32)
        store = InMemoryShardStore.from_vectors(ids, contents, vectors)
        centroid = _normalize(vectors.mean(axis=0, keepdims=True))[0]
        shards.append(Shard(id=shard_name, store=store, cost=SHARD_COSTS[shard_name], centroid=centroid))
    return shards


@dataclass
class StrategyRun:
    strategy: str
    judge: str
    hit_at_k: float
    mrr: float
    avg_shards_contacted: float


def _hit_and_rank(expected_id: str, result_ids: list[str]) -> tuple[bool, float]:
    if expected_id in result_ids:
        rank = result_ids.index(expected_id) + 1
        return True, 1.0 / rank
    return False, 0.0


def run_sweep(
    shards: list[Shard],
    task_vectors: dict[str, np.ndarray],
    k: int,
    llm_judge: bool,
    llm_judge_tasks: int,
) -> list[StrategyRun]:
    geometric = GeometricJudge()
    claude_cli = ClaudeCLIJudge() if llm_judge else None

    rows: list[StrategyRun] = []
    for strategy_name, strategy_fn in STRATEGIES.items():
        judges_to_run: list[tuple[str, object, int]] = [("geometric", geometric, len(TASKS))]
        if llm_judge and strategy_name in JUDGE_SENSITIVE_STRATEGIES:
            judges_to_run.append(("claude_cli", claude_cli, min(llm_judge_tasks, len(TASKS))))

        for judge_name, judge, n_tasks in judges_to_run:
            hits, reciprocal_ranks, contacted = [], [], []
            for task in TASKS[:n_tasks]:
                query_vec = task_vectors[task.query]
                trace = strategy_fn(
                    query=task.query,
                    query_vec=query_vec,
                    shards=shards,
                    budget=GENEROUS_BUDGET,
                    k=k,
                    judge=judge,
                )
                result_ids = [r.id for r in trace.results]
                hit, rr = _hit_and_rank(task.expected_id, result_ids)
                hits.append(hit)
                reciprocal_ranks.append(rr)
                contacted.append(len(trace.shards_contacted))

            rows.append(
                StrategyRun(
                    strategy=strategy_name,
                    judge=judge_name,
                    hit_at_k=float(np.mean(hits)),
                    mrr=float(np.mean(reciprocal_ranks)),
                    avg_shards_contacted=float(np.mean(contacted)),
                )
            )
            print(
                f"  {strategy_name:24s} [{judge_name:9s}] hit@{k}={rows[-1].hit_at_k:.2f}  "
                f"MRR={rows[-1].mrr:.2f}  shards={rows[-1].avg_shards_contacted:.1f}/3  (n={n_tasks})"
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--llm-judge-tasks", type=int, default=8)
    args = parser.parse_args()

    memory_texts = [m.content for m in MEMORIES]
    memory_ids = [m.id for m in MEMORIES]
    memory_vecs = embed_all(memory_texts, cache_key="agent_tasks_memories")
    memory_vec_by_id = dict(zip(memory_ids, memory_vecs))

    task_queries = [t.query for t in TASKS]
    task_vecs = embed_all(task_queries, cache_key="agent_tasks_queries")
    task_vec_by_query = dict(zip(task_queries, task_vecs))

    shards = build_shards(memory_vec_by_id)

    print(f"=== agent_tasks: {len(TASKS)} hand-labeled tasks, {len(MEMORIES)} memories, k={args.top_k} ===\n")
    rows = run_sweep(shards, task_vec_by_query, k=args.top_k, llm_judge=args.llm_judge, llm_judge_tasks=args.llm_judge_tasks)

    print("\n=== Headline: geometric judge, hit@k by strategy ===")
    for row in rows:
        if row.judge == "geometric":
            print(f"  {row.strategy:24s} hit@{args.top_k}={row.hit_at_k:.2f}  shards={row.avg_shards_contacted:.1f}/3")

    llm_rows = [r for r in rows if r.judge == "claude_cli"]
    if llm_rows:
        print("\n=== geometric vs. LLM judge (judge-sensitive strategies only) ===")
        for llm_row in llm_rows:
            geo_row = next(r for r in rows if r.judge == "geometric" and r.strategy == llm_row.strategy)
            print(
                f"  {llm_row.strategy:24s} geometric: hit@{args.top_k}={geo_row.hit_at_k:.2f} shards={geo_row.avg_shards_contacted:.1f}  |  "
                f"llm: hit@{args.top_k}={llm_row.hit_at_k:.2f} shards={llm_row.avg_shards_contacted:.1f}"
            )


if __name__ == "__main__":
    main()
