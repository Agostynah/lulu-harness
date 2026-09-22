"""evals/agent_tasks/run.py: the geometric-vs-LLM-judge comparison
evals/dbpedia can't actually provide, because DBpedia14's "queries" are
just other documents used as similarity probes, not real task questions
(see evals/dbpedia/run.py's docstring). These hand-labeled tasks
(tasks.py) against a hand-built personal memory bank (memories.py) are
real natural-language questions -- recall here means "did the router
surface the memory a human actually wanted," not "did it find the
nearest vector."

Shards are partitioned by type (semantic/procedural/episodic), matching
memory.py's own design -- not KMeans, which is what a 100K-document
research corpus needs, not a ~150-memory personal bank.

Usage:
    uv run python evals/agent_tasks/run.py
    uv run python evals/agent_tasks/run.py --llm-judge --llm-judge-tasks 126
    uv run python evals/agent_tasks/run.py --jev-judge   # needs JEV_API_KEY
    uv run python evals/agent_tasks/run.py --jev-fallback-test

--jev-judge runs the real Jev API against all tasks (not subsampled like
--llm-judge -- Jev's whole pitch is being cheap/fast enough that it
doesn't need one; see judges/jev.py). --jev-fallback-test answers a
different question: does FallbackJudge's safety net actually work at
eval scale, not just in the mocked unit tests (judges/test_judges.py)?
It wraps a JevJudge given a deliberately invalid key -- a REAL failed API
call, not a mock -- in FallbackJudge and asserts the results come out
identical to pure GeometricJudge.

Every judge call is individually timed (TimedJudge) -- shards-contacted
alone is a downstream memory-search cost, not the judge's own call cost,
and a judge that saves shards but is itself slow could easily be a net
loss end to end. Paired McNemar tests check whether a hit-rate
difference between two judges on the SAME tasks is real or could
plausibly be noise. Full per-task results, latencies, and test
statistics are written to results.json for downstream analysis/plotting
-- the printed tables are a summary of that file, not the source of
truth.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Any

import numpy as np
from lulu_router.backends.memory import InMemoryShardStore
from lulu_router.cost import Budget, CostProfile
from lulu_router.judges.claude_cli import ClaudeCLIJudge
from lulu_router.judges.fallback import FallbackJudge
from lulu_router.judges.geometric import GeometricJudge
from lulu_router.judges.jev import JevJudge
from lulu_router.shard import Shard
from lulu_router.strategies import STRATEGIES

from lulu.config import load_dotenv

from memories import MEMORIES
from tasks import TASKS

REPO_ROOT = Path(__file__).parent.parent.parent

CACHE_DIR = Path(__file__).parent / ".cache"
RESULTS_PATH = Path(__file__).parent / "results.json"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
TOP_K = 3

SHARD_COSTS = {
    "episodic": CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=80),
    "semantic": CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=120),
    "procedural": CostProfile(latency_ms=5.0, usd_per_query=0.0, tokens_per_result=150),
}

GENEROUS_BUDGET = Budget(max_tokens=100_000, max_latency_ms=60_000.0, max_usd=10.0)

# Same reasoning as evals/dbpedia: only strategies whose routing decision
# actually depends on the judge's verdict are worth re-running under a
# judge other than geometric.
JUDGE_SENSITIVE_STRATEGIES = ("confidence_threshold", "progressive_expansion")


class TimedJudge:
    """Wraps any Judge and records the wall-clock duration of every
    judge() call. Answers a question shards-contacted alone can't:
    geometric is ~free local math, Jev is a real network call, and the
    LLM judge is an even heavier one (spawns a whole `claude -p`
    process) -- a judge that saves shards but is itself slow could
    still be a net loss end to end, and that can't be seen without
    actually timing the calls."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.name = inner.name
        self.call_latencies_ms: list[float] = []

    def judge(self, *args, **kwargs):
        start = time.perf_counter()
        result = self.inner.judge(*args, **kwargs)
        self.call_latencies_ms.append((time.perf_counter() - start) * 1000.0)
        return result


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
class TaskResult:
    query: str
    expected_id: str
    hit: bool
    reciprocal_rank: float
    shards_contacted: int


@dataclass
class StrategyRun:
    strategy: str
    judge: str
    hit_at_k: float
    mrr: float
    avg_shards_contacted: float
    judge_latency_ms: dict[str, float]  # mean/p50/p95/n, or {} if not timed
    per_task: list[TaskResult] = field(default_factory=list)


def _hit_and_rank(expected_id: str, result_ids: list[str]) -> tuple[bool, float]:
    if expected_id in result_ids:
        rank = result_ids.index(expected_id) + 1
        return True, 1.0 / rank
    return False, 0.0


def _latency_stats(latencies_ms: list[float]) -> dict[str, float]:
    if not latencies_ms:
        return {}
    arr = np.array(latencies_ms)
    return {
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "n_calls": int(len(arr)),
    }


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar test p-value: on paired binary outcomes
    (same tasks, two judges), b = judge A right & B wrong, c = the
    reverse. Only the discordant pairs (n = b + c) carry information --
    both judges agreeing, right or wrong, says nothing about which is
    better. Tests the discordant pairs against a fair coin (p=0.5);
    exact binomial rather than the usual chi-square approximation, which
    is unreliable once n gets small (some of these strategy/judge
    comparisons have well under 20 discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p_le_k = sum(comb(n, i) * (0.5**n) for i in range(k + 1))
    return min(1.0, 2 * p_le_k)


def mcnemar_between(rows: list[StrategyRun], strategy: str, judge_a: str, judge_b: str) -> dict[str, Any] | None:
    row_a = next((r for r in rows if r.strategy == strategy and r.judge == judge_a), None)
    row_b = next((r for r in rows if r.strategy == strategy and r.judge == judge_b), None)
    if row_a is None or row_b is None or len(row_a.per_task) != len(row_b.per_task):
        return None
    b = sum(1 for ta, tb in zip(row_a.per_task, row_b.per_task) if ta.hit and not tb.hit)
    c = sum(1 for ta, tb in zip(row_a.per_task, row_b.per_task) if not ta.hit and tb.hit)
    return {
        "strategy": strategy,
        "judge_a": judge_a,
        "judge_b": judge_b,
        "n_tasks": len(row_a.per_task),
        "a_right_b_wrong": b,
        "b_right_a_wrong": c,
        "p_value": exact_mcnemar_p(b, c),
        "significant_at_0.05": exact_mcnemar_p(b, c) < 0.05,
    }


def run_sweep(
    shards: list[Shard],
    task_vectors: dict[str, np.ndarray],
    k: int,
    llm_judge: bool,
    llm_judge_tasks: int,
    jev_judge: bool,
    jev_fallback_test: bool,
) -> list[StrategyRun]:
    geometric = TimedJudge(GeometricJudge())
    claude_cli = TimedJudge(ClaudeCLIJudge()) if llm_judge else None
    # Real API, not subsampled like claude_cli -- Jev's own pitch
    # (judges/jev.py: "20-200x faster, 40-1000x cheaper" than a full LLM
    # call) is exactly what makes that affordable here.
    jev = TimedJudge(JevJudge(api_key=os.environ["JEV_API_KEY"])) if jev_judge else None
    # A deliberately invalid key -- a REAL failed API call (401), not a
    # mock -- wrapped in the actual FallbackJudge production code uses.
    jev_fallback = (
        TimedJudge(FallbackJudge(primary=JevJudge(api_key="invalid-key-for-fallback-test"), secondary=GeometricJudge()))
        if jev_fallback_test
        else None
    )

    rows: list[StrategyRun] = []
    for strategy_name, strategy_fn in STRATEGIES.items():
        judges_to_run: list[tuple[str, object, int]] = [("geometric", geometric, len(TASKS))]
        if llm_judge and strategy_name in JUDGE_SENSITIVE_STRATEGIES:
            judges_to_run.append(("claude_cli", claude_cli, min(llm_judge_tasks, len(TASKS))))
        if jev_judge and strategy_name in JUDGE_SENSITIVE_STRATEGIES:
            judges_to_run.append(("jev", jev, len(TASKS)))
        if jev_fallback_test and strategy_name in JUDGE_SENSITIVE_STRATEGIES:
            judges_to_run.append(("jev_fallback", jev_fallback, len(TASKS)))

        for judge_name, judge, n_tasks in judges_to_run:
            judge.call_latencies_ms.clear()
            per_task: list[TaskResult] = []
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
                per_task.append(
                    TaskResult(
                        query=task.query,
                        expected_id=task.expected_id,
                        hit=hit,
                        reciprocal_rank=rr,
                        shards_contacted=len(trace.shards_contacted),
                    )
                )

            latency_stats = _latency_stats(list(judge.call_latencies_ms))
            rows.append(
                StrategyRun(
                    strategy=strategy_name,
                    judge=judge_name,
                    hit_at_k=float(np.mean([t.hit for t in per_task])),
                    mrr=float(np.mean([t.reciprocal_rank for t in per_task])),
                    avg_shards_contacted=float(np.mean([t.shards_contacted for t in per_task])),
                    judge_latency_ms=latency_stats,
                    per_task=per_task,
                )
            )
            latency_str = f"  judge_latency_p50={latency_stats['p50_ms']:.1f}ms" if latency_stats else ""
            print(
                f"  {strategy_name:24s} [{judge_name:9s}] hit@{k}={rows[-1].hit_at_k:.2f}  "
                f"MRR={rows[-1].mrr:.2f}  shards={rows[-1].avg_shards_contacted:.1f}/3  (n={n_tasks}){latency_str}"
            )
    return rows


def build_report(rows: list[StrategyRun], top_k: int) -> dict[str, Any]:
    judges_present = sorted({r.judge for r in rows})
    significance = []
    for strategy in JUDGE_SENSITIVE_STRATEGIES:
        strategy_judges = [j for j in judges_present if j != "geometric"]
        for other in strategy_judges:
            result = mcnemar_between(rows, strategy, "geometric", other)
            if result:
                significance.append(result)

    return {
        "meta": {
            "n_tasks": len(TASKS),
            "n_memories": len(MEMORIES),
            "top_k": top_k,
            "embedding_model": EMBEDDING_MODEL,
            "judges_present": judges_present,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        },
        "per_strategy": [
            {
                "strategy": r.strategy,
                "judge": r.judge,
                "n_tasks": len(r.per_task),
                "hit_at_k": r.hit_at_k,
                "mrr": r.mrr,
                "avg_shards_contacted": r.avg_shards_contacted,
                "judge_latency_ms": r.judge_latency_ms,
                "per_task": [
                    {
                        "query": t.query,
                        "expected_id": t.expected_id,
                        "hit": t.hit,
                        "reciprocal_rank": t.reciprocal_rank,
                        "shards_contacted": t.shards_contacted,
                    }
                    for t in r.per_task
                ],
            }
            for r in rows
        ],
        "significance_tests": significance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--llm-judge", action="store_true")
    parser.add_argument("--llm-judge-tasks", type=int, default=8)
    parser.add_argument("--jev-judge", action="store_true", help="needs JEV_API_KEY (.env or exported)")
    parser.add_argument(
        "--jev-fallback-test",
        action="store_true",
        help="proves FallbackJudge recovers a real (not mocked) Jev failure -- no key needed",
    )
    parser.add_argument(
        "--no-write-results",
        action="store_true",
        help="skip writing results.json (default: always write it)",
    )
    args = parser.parse_args()

    load_dotenv(REPO_ROOT)
    if args.jev_judge and not os.environ.get("JEV_API_KEY"):
        parser.error("--jev-judge needs JEV_API_KEY set (in .env or the environment)")

    memory_texts = [m.content for m in MEMORIES]
    memory_ids = [m.id for m in MEMORIES]
    memory_vecs = embed_all(memory_texts, cache_key="agent_tasks_memories")
    memory_vec_by_id = dict(zip(memory_ids, memory_vecs))

    task_queries = [t.query for t in TASKS]
    task_vecs = embed_all(task_queries, cache_key="agent_tasks_queries")
    task_vec_by_query = dict(zip(task_queries, task_vecs))

    shards = build_shards(memory_vec_by_id)

    print(f"=== agent_tasks: {len(TASKS)} hand-labeled tasks, {len(MEMORIES)} memories, k={args.top_k} ===\n")
    rows = run_sweep(
        shards,
        task_vec_by_query,
        k=args.top_k,
        llm_judge=args.llm_judge,
        llm_judge_tasks=args.llm_judge_tasks,
        jev_judge=args.jev_judge,
        jev_fallback_test=args.jev_fallback_test,
    )

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

    jev_rows = [r for r in rows if r.judge == "jev"]
    if jev_rows:
        print(f"\n=== geometric vs. Jev (judge-sensitive strategies, all {len(TASKS)} tasks) ===")
        for jev_row in jev_rows:
            geo_row = next(r for r in rows if r.judge == "geometric" and r.strategy == jev_row.strategy)
            print(
                f"  {jev_row.strategy:24s} geometric: hit@{args.top_k}={geo_row.hit_at_k:.2f} MRR={geo_row.mrr:.2f} shards={geo_row.avg_shards_contacted:.1f}  |  "
                f"jev: hit@{args.top_k}={jev_row.hit_at_k:.2f} MRR={jev_row.mrr:.2f} shards={jev_row.avg_shards_contacted:.1f}"
            )

    fallback_rows = [r for r in rows if r.judge == "jev_fallback"]
    if fallback_rows:
        print("\n=== FallbackJudge recovery check (real invalid-key failure, not mocked) ===")
        all_matched = True
        for fb_row in fallback_rows:
            geo_row = next(r for r in rows if r.judge == "geometric" and r.strategy == fb_row.strategy)
            matched = fb_row.hit_at_k == geo_row.hit_at_k and fb_row.avg_shards_contacted == geo_row.avg_shards_contacted
            all_matched = all_matched and matched
            print(
                f"  {fb_row.strategy:24s} geometric: hit@{args.top_k}={geo_row.hit_at_k:.2f} shards={geo_row.avg_shards_contacted:.1f}  |  "
                f"jev_fallback: hit@{args.top_k}={fb_row.hit_at_k:.2f} shards={fb_row.avg_shards_contacted:.1f}  "
                f"[{'MATCH' if matched else 'MISMATCH -- fallback is NOT behaving identically to geometric'}]"
            )
        print(
            f"\n  {'PASS' if all_matched else 'FAIL'}: FallbackJudge under a real Jev failure "
            f"{'exactly reproduces' if all_matched else 'DOES NOT reproduce'} plain GeometricJudge."
        )

    report = build_report(rows, args.top_k)

    if report["significance_tests"]:
        print("\n=== Significance (exact McNemar, geometric vs. other judges, judge-sensitive strategies) ===")
        for test in report["significance_tests"]:
            sig = "SIGNIFICANT (p<0.05)" if test["significant_at_0.05"] else "not significant"
            print(
                f"  {test['strategy']:24s} geometric vs {test['judge_b']:10s} "
                f"discordant: {test['a_right_b_wrong']} vs {test['b_right_a_wrong']}  "
                f"p={test['p_value']:.4f}  [{sig}]"
            )

    latency_rows = [r for r in rows if r.judge_latency_ms]
    if latency_rows:
        print("\n=== Judge call latency (own cost, separate from shards-contacted) ===")
        for r in latency_rows:
            s = r.judge_latency_ms
            print(f"  [{r.judge:9s}] {r.strategy:24s} mean={s['mean_ms']:.1f}ms  p50={s['p50_ms']:.1f}ms  p95={s['p95_ms']:.1f}ms  n={s['n_calls']}")

    if not args.no_write_results:
        RESULTS_PATH.write_text(json.dumps(report, indent=2))
        print(f"\nFull results (per-task, latencies, significance tests) written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
