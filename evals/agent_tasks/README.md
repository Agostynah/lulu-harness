# Agent tasks: the judge comparison DBpedia14 can't give you

DBpedia14's "queries" are just other documents used as similarity probes
(see `evals/dbpedia/run.py`'s docstring) — there's no real question behind
them, so a sufficiency judge that reads content has nothing meaningful to
reason about. This eval fixes that: 25 hand-labeled, natural-language
tasks (`tasks.py`) against an 18-memory hand-built project memory bank
(`memories.py`) — 5 semantic/decisions, 5 procedural/runbooks, 8
episodic/bug-fixes, with ~7 deliberate paraphrases mixed in to test
semantic matching, not just keyword overlap. Recall here means "did the
router surface the memory a human actually wanted," not "did it find the
nearest vector."

Shards are partitioned by type (episodic/semantic/procedural), matching
`memory.py`'s own design — not KMeans, which is what a 100K-document
research corpus needs, not an 18-memory project bank.

```bash
# geometric judge only, all 6 strategies, all 25 tasks -- fast, no API calls
uv run python evals/agent_tasks/run.py

# add the LLM-judge axis (shells out to `claude -p --model haiku` per
# judgment, so kept to a small subset of tasks by default)
uv run python evals/agent_tasks/run.py --llm-judge --llm-judge-tasks 8
```

## Geometric judge, all 25 tasks, k=3

| strategy | hit@3 | MRR | shards contacted |
|---|---|---|---|
| query_all | 1.00 | 1.00 | 3.0/3 |
| flat_topk | 1.00 | 1.00 | 3.0/3 |
| top_n_neighbors | 1.00 | 1.00 | 3.0/3 |
| confidence_threshold | 1.00 | 1.00 | 2.8/3 |
| progressive_expansion | 1.00 | 1.00 | 2.8/3 |
| budgeted_communication | 1.00 | 1.00 | 3.0/3 |

At this scale (3 shards, 25 tasks) every strategy hits perfect recall —
there just isn't enough shard diversity for routing to meaningfully skip
one and still find the answer. The one real signal here:
`confidence_threshold`/`progressive_expansion` already skip ~7% of shard
contacts (2.8/3) at zero recall cost, for free, purely from the
geometric judge being confident enough not to check the third shard.

## Geometric vs. LLM judge (8-task subset, real API calls)

The comparison this eval exists for. On real task queries — not DBpedia's
similarity probes — the LLM judge reads candidate content and decides
"does this actually answer the question," instead of just measuring
vector-space gap. Real run, `--llm-judge --llm-judge-tasks 8`:

| strategy | judge | hit@3 | shards contacted |
|---|---|---|---|
| confidence_threshold | geometric | 1.00 | 2.8/3 |
| confidence_threshold | LLM (`ClaudeCLIJudge`, haiku) | 0.88 | 1.1/3 |
| progressive_expansion | geometric | 1.00 | 2.8/3 |
| progressive_expansion | LLM (`ClaudeCLIJudge`, haiku) | 1.00 | 1.1/3 |

The LLM judge matches or nearly matches geometric recall (1.00/0.88 vs.
1.00) while contacting a bit over a third as many shards (1.1/3 vs.
2.8/3) — it's willing to stop after one shard once it's actually read
enough to answer, where the geometric judge keeps checking based on
vector-space confidence alone regardless of what the content says.
`confidence_threshold` gives up one task's recall for that;
`progressive_expansion` gets the same cost win for free. That's a genuine,
measurable quality/cost trade-off on real task queries, which is exactly
what DBpedia14 (see `evals/dbpedia/README.md`) structurally cannot show:
its "LLM-judge plumbing check" section demonstrates the wiring works, not
whether the judge is actually good, because its queries have no real
sufficiency question to answer.

Numbers are inherently a bit noisy at n=8 (a single LLM judgment can flip
the recall count by 0.125) — re-run with a larger `--llm-judge-tasks` for
a tighter estimate; it costs one `claude -p` call per task per
judge-sensitive strategy.
