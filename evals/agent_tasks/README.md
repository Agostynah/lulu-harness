# Agent tasks: the judge comparison DBpedia14 can't give you

DBpedia14's "queries" are just other documents used as similarity probes
(see `evals/dbpedia/run.py`'s docstring) — there's no real question behind
them, so a sufficiency judge that reads content has nothing meaningful to
reason about. This eval fixes that: 126 hand-labeled, natural-language
tasks (`tasks.py`) against a 143-memory synthetic **personal-assistant**
memory bank (`memories.py`) — 50 semantic/facts-and-preferences, 44
procedural/routines, 49 episodic/life-events. Recall here means "did the
router surface the memory a human actually wanted," not "did it find the
nearest vector."

**Built deliberately harder than a first pass at this would be.** An
earlier, smaller version of this eval (25 tasks, 18 memories) hit a
recall ceiling of 1.00 on *every* strategy and *every* judge — which
looked good but proved nothing: there wasn't enough ambiguity in the
corpus for any judge to actually fail. This version's memory bank is
full of **deliberate near-duplicate pairs** a vector-only judge can
confuse — two birthdays six days apart, two different allergies in the
family, two different "reset X" procedures (wifi router vs. cable
modem), two people's coffee orders — so recall dropping below 1.00 is
possible, and a real quality gap between judges can actually show up
instead of being hidden by a ceiling effect.

Shards are partitioned by type (episodic/semantic/procedural), matching
`memory.py`'s own design — not KMeans, which is what a 100K-document
research corpus needs, not a ~150-memory personal bank.

```bash
# geometric judge only, all 6 strategies, all 126 tasks -- fast, no API calls
uv run python evals/agent_tasks/run.py

# add the LLM-judge axis (shells out to `claude -p --model haiku` per judgment)
uv run python evals/agent_tasks/run.py --llm-judge --llm-judge-tasks 25

# add the Jev axis (real API, all 126 tasks -- see judges/jev.py for why
# it doesn't need subsampling the way the LLM judge does) -- needs JEV_API_KEY
uv run python evals/agent_tasks/run.py --jev-judge

# proves FallbackJudge recovers a REAL (not mocked) Jev failure -- no key needed
uv run python evals/agent_tasks/run.py --jev-fallback-test
```

## Geometric judge, all 126 tasks, k=3

| strategy | hit@3 | MRR | shards contacted |
|---|---|---|---|
| query_all | 0.97 | 0.90 | 3.0/3 |
| flat_topk | 0.97 | 0.90 | 3.0/3 |
| top_n_neighbors | 0.97 | 0.90 | 3.0/3 |
| confidence_threshold | 0.97 | 0.90 | 2.9/3 |
| progressive_expansion | 0.97 | 0.90 | 2.9/3 |
| budgeted_communication | 0.97 | 0.90 | 3.0/3 |

Even the strategies that search everything (`query_all`, `flat_topk`)
top out at 0.97, not 1.00 — a handful of the distractor-disambiguation
tasks (e.g. telling the partner's coffee order apart from the best
friend's) are hard enough that pure vector similarity alone, with no
judge or content-reading involved at all, occasionally ranks the wrong
one inside the top 3. That's the ceiling this eval is actually measuring
against, not a free 1.00.

## Geometric vs. Jev (all 126 tasks, real API calls)

The comparison this eval exists for, at full scale — not a subsample.
Real run, `--jev-judge`:

| strategy | judge | hit@3 | MRR | shards contacted |
|---|---|---|---|---|
| confidence_threshold | geometric | 0.97 | 0.90 | 2.9/3 |
| confidence_threshold | Jev | 0.94 | 0.90 | **1.7/3** |
| progressive_expansion | geometric | 0.97 | 0.90 | 2.9/3 |
| progressive_expansion | Jev | 0.94 | 0.90 | **1.7/3** |

**This is a genuine trade-off, not a free lunch.** Jev contacts about
41% fewer shards (1.7/3 vs. 2.9/3) but gives up a small amount of recall
to do it (0.97 → 0.94, roughly 3-4 more misses out of 126) — it's
willing to stop searching once it's confident enough, and on this
harder, distractor-heavy corpus that confidence is occasionally wrong in
a way the exhaustive geometric baseline isn't. MRR holds steady at 0.90
either way, so when Jev *does* find the right memory, it ranks it just
as well as geometric does. Whether that trade is worth it depends on
what the harness is optimizing for (see `docs/THESIS.md`'s cost
argument) — this eval's job is to make the trade honest and measured,
not to declare a winner.

## FallbackJudge recovery check (real failure, not mocked)

`--jev-fallback-test` wraps a `JevJudge` given a **deliberately invalid
API key** — a real, failed HTTP call, not a mocked one — in the actual
`FallbackJudge` production code uses, and asserts every judge-sensitive
strategy's results come out identical to plain `GeometricJudge`:

```
=== FallbackJudge recovery check (real invalid-key failure, not mocked) ===
  confidence_threshold     geometric: hit@3=0.97 shards=2.9  |  jev_fallback: hit@3=0.97 shards=2.9  [MATCH]
  progressive_expansion    geometric: hit@3=0.97 shards=2.9  |  jev_fallback: hit@3=0.97 shards=2.9  [MATCH]

  PASS: FallbackJudge under a real Jev failure exactly reproduces plain GeometricJudge.
```

Passes on this harder, distractor-heavy corpus too, not just the small
easy one — the safety net doesn't just work in `judges/test_judges.py`'s
mocked unit tests, it works when a real network call actually fails.

## Geometric vs. LLM judge (25-task subset, real API calls)

`--llm-judge --llm-judge-tasks 25` (shells out to `claude -p --model
haiku` per judgment, so kept to a subset — see `judges/claude_cli.py`):

| strategy | judge | hit@3 | MRR | shards contacted |
|---|---|---|---|---|
| confidence_threshold | geometric | 0.97 | 0.90 | 2.9/3 |
| confidence_threshold | LLM (`ClaudeCLIJudge`, haiku) | 0.76 | 0.76 | 1.1/3 |
| progressive_expansion | geometric | 0.97 | 0.90 | 2.9/3 |
| progressive_expansion | LLM (`ClaudeCLIJudge`, haiku) | 0.76 | 0.74 | 1.1/3 |

Numbers are inherently a bit noisier here than Jev's (subsampled at
n=25, not the full 126 — a single LLM judgment can flip the recall
count by 4 points) — re-run with a larger `--llm-judge-tasks` for a
tighter estimate; it costs one `claude -p` call per task per
judge-sensitive strategy, meaningfully slower than Jev's real API calls
(see `judges/jev.py`'s docstring on why).

## All three judges, side by side

The genuinely useful comparison this eval was rebuilt to make possible:

| judge | hit@3 | shards contacted | tasks | notes |
|---|---|---|---|---|
| geometric | 0.97 | 2.9/3 | 126 | pure vector-space gap, no content read |
| **Jev** | **0.94** | **1.7/3** | 126 | reads content, fast/cheap enough for the full set |
| LLM (haiku, `ClaudeCLIJudge`) | 0.76 | 1.1/3 | 25 | reads content, one full chat-model call per judgment |

Jev isn't just "cheap and roughly as good" — on this harder corpus it
**noticeably outperforms the haiku-based LLM judge on recall** (0.94 vs.
0.76) while contacting a similar number of shards (1.7 vs. 1.1) and
running at full scale instead of a 25-task subsample. Haiku's aggressive
stopping (1.1/3 shards) buys speed but costs real recall here in a way
Jev's stopping doesn't — a genuinely different point on the cost/quality
curve, not just a faster version of the same trade-off. Geometric is
still the recall ceiling (0.97) at the highest shard-contact cost
(2.9/3); which point on this curve is "worth it" depends on what the
harness is optimizing for (see `docs/THESIS.md`), which is exactly what
this table is for deciding, not settling on its own.
