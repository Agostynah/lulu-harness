# Agent tasks: the judge comparison DBpedia14 can't give you

DBpedia14's "queries" are just other documents used as similarity probes
(see `evals/dbpedia/run.py`'s docstring) — there's no real question
behind them, so a sufficiency judge that reads content has nothing
meaningful to reason about. This eval fixes that: hand-labeled,
natural-language tasks (`tasks.py`) against a synthetic
**personal-assistant** memory bank (`memories.py`) — currently 209 tasks
against 220 memories (90 semantic/facts-and-preferences, 65
procedural/routines, 65 episodic/life-events). Recall here means "did
the router surface the memory a human actually wanted," not "did it
find the nearest vector."

**→ [METHODOLOGY.md](METHODOLOGY.md) is the full writeup** — hypothesis,
method, every metric and what it does/doesn't measure, statistical
significance testing, the complete results (recall, MRR, shard cost,
real judge-call latency), and honest limitations. This file is just the
quickstart; don't duplicate numbers here that live there.

**Built deliberately harder than a naive first pass.** An earlier,
smaller version of this eval (25 tasks, 18 memories) hit a recall
ceiling of 1.00 on *every* strategy and *every* judge — which looked
good but proved nothing: there wasn't enough ambiguity in the corpus for
any judge to actually fail. This version's memory bank is full of
**deliberate near-duplicate distractor pairs** a vector-only judge can
confuse — two birthdays six days apart, two different allergies in the
family, two different "reset X" procedures (wifi router vs. cable
modem), whose gadget warranty is active vs. expired — so even the
exhaustive baseline strategies now top out at 0.97 recall, not 1.00.
That's a real ceiling to measure against.

Shards are partitioned by type (episodic/semantic/procedural), matching
`memory.py`'s own design — not KMeans, which is what a 100K-document
research corpus needs, not a ~220-memory personal bank.

```bash
# geometric judge only, all 6 strategies -- fast, no API calls
uv run python evals/agent_tasks/run.py

# + Jev, real API calls, full corpus (no subsampling -- see judges/jev.py)
uv run python evals/agent_tasks/run.py --jev-judge          # needs JEV_API_KEY

# proves FallbackJudge recovers a REAL (not mocked) Jev failure -- no key needed
uv run python evals/agent_tasks/run.py --jev-fallback-test

# + the LLM judge (shells out to `claude -p --model haiku` per judgment --
# slow and a real $ cost, so pass a task count deliberately)
uv run python evals/agent_tasks/run.py --llm-judge --llm-judge-tasks 126
```

Every run overwrites `results.json` with full per-task data (hits,
ranks, shard counts, judge call latencies) for whatever judges were
run — that's the machine-readable source `METHODOLOGY.md`'s tables were
built from. `results_n126_all_judges.json` and
`results_n209_geometric_vs_jev.json` are point-in-time snapshots kept
alongside it for reproducibility of the specific numbers cited in the
writeup, since `results.json` itself gets overwritten by the next run.
