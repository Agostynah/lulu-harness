# Is Jev actually worth it? A test report

**Question:** does routing memory-search decisions through Jev (TypeSafe
AI's content-reading judge) actually add value to Lulu's memory router,
compared to the existing `GeometricJudge` baseline it falls back to?

**Short answer:** partially, and with a real cost the earlier, smaller
version of this test hid. Jev contacts ~41% fewer memory shards than
geometric for the same routing decision, but that's a **cheaper**
decision, not a **faster** one -- Jev's own API call takes longer than
the shards it saves would have cost. It also gives up a small amount of
recall, one that's right at the edge of statistical significance rather
than clearly proven or clearly noise. `FallbackJudge`'s safety net --
falling back to geometric on a real (not mocked) Jev failure -- works
exactly as designed, with no exceptions found.

This document is the full writeup: hypothesis, method, every metric
used and what each one actually measures, the statistical test, the
raw results, and the honest limitations. Machine-readable versions of
every table here are in `results_n126_all_judges.json` (4-judge
comparison, 126 tasks) and `results_n209_geometric_vs_jev.json`
(geometric vs. Jev only, 209 tasks, run without the other two judges
specifically to afford more tasks without re-paying for `claude -p`
calls) -- generated straight from `run.py`, not hand-transcribed.

---

## 1. Hypothesis

Jev is pitched (see the root README and `judges/jev.py`) as a
"System 1" evaluation model: fast and cheap enough to run on every
routing decision, reading actual shard content instead of just
vector-space geometry (`GeometricJudge`'s method), and therefore able to
stop searching earlier than geometric once it's actually confident the
retrieved content answers the query -- without giving up much recall to
do it.

That's a claim with three separable, testable parts:

1. **Cost**: Jev-driven routing contacts fewer memory shards than
   geometric for the same task, on average.
2. **Quality**: Jev-driven routing doesn't meaningfully sacrifice recall
   (finding the memory the user actually wanted) to get that cost win.
3. **Reliability**: when Jev itself is unavailable, `FallbackJudge`
   recovers geometric's behavior exactly, so adopting Jev doesn't
   introduce a new failure mode into the harness.

Each is tested separately below.

## 2. Why the first version of this test was thrown out

An earlier pass at this eval (18 memories, 25 tasks, 3 shards) reported
Jev matching geometric's recall (1.00 vs. 1.00) at a third of the shard
cost. That result was true and also **worthless as evidence**: every
judge, including the trivial ones, hit 1.00 recall on that corpus,
because there wasn't enough ambiguity in 18 memories across 3 shards for
routing to ever actually risk missing the right answer. A test that
can't fail can't tell you anything about quality.

This version's memory bank (`memories.py`) was rebuilt specifically to
contain **deliberate near-duplicate distractor pairs** -- two people's
birthdays six days apart, two different family members' allergies, two
different "reset X" procedures (wifi router vs. cable modem), two
people's coffee orders, which gadget's warranty is active vs. expired,
and so on. With those in place, even the *exhaustive* baseline strategies
(`query_all`, `flat_topk` -- which search every shard, no judge
involved) only reach 0.97 recall, not 1.00. That's the real ceiling this
test measures against.

## 3. Method

**Corpus**: `memories.py`, 220 synthetic personal-assistant memories
across three shards matching `lulu.memory`'s own partitioning --
`semantic` (facts/preferences/decisions, 90 memories), `procedural`
(routines/how-tos, 65 memories), `episodic` (life events, 65 memories).
Built in two waves; wave 2 added specifically to increase statistical
power for the geometric-vs-Jev comparison (see §6).

**Tasks**: `tasks.py`, 209 hand-labeled natural-language questions, each
with exactly one correct expected memory id. A large minority are
distractor-disambiguation tasks targeting a specific near-duplicate pair
-- e.g. "Whose warranty is still active, my laptop or my phone?" has
only one correct answer even though both "laptop warranty" and "phone
warranty" memories exist and are close in vector space.

**Router setup**: the same `lulu_router` code the harness runs in
production -- `MemoryRouter` dispatching to all 6 `STRATEGIES`, `k=3`, a
generous budget (no strategy is budget-constrained in this test; the
judge's own stopping decision is what's being measured, not budget
enforcement). Only `confidence_threshold` and `progressive_expansion`
have routing decisions that actually depend on the judge's verdict
(`JUDGE_SENSITIVE_STRATEGIES`) -- the other four either search
everything or use a fixed neighbor count regardless of judge, so judge
identity can't change their outcome and re-running them per judge would
just be wasted API calls.

**Judges compared**: `GeometricJudge` (baseline, pure vector-space score
gap, no network call), `JevJudge` (TypeSafe AI's Jev, real API calls,
`JEV_API_KEY` required), `ClaudeCLIJudge` (shells out to `claude -p
--model haiku` per judgment), and `FallbackJudge` wrapping a `JevJudge`
constructed with a **deliberately invalid API key** -- a real failed
HTTP call, not a mock -- to test the actual recovery path.

**Reproduction**:
```bash
uv run python evals/agent_tasks/run.py                              # geometric only, instant
uv run python evals/agent_tasks/run.py --jev-judge                  # + Jev, needs JEV_API_KEY
uv run python evals/agent_tasks/run.py --jev-fallback-test           # FallbackJudge recovery check
uv run python evals/agent_tasks/run.py --llm-judge --llm-judge-tasks 126   # + haiku (slow, real $ cost)
```
Every run overwrites `results.json` with the full per-task data for
whatever combination of flags was passed and prints the same tables
shown below. `--jev-judge` was deliberately run *without* `--llm-judge`
for the 209-task expansion, on request, so as not to spend more real
`claude -p` calls once the recall/latency picture for haiku was already
clear from the 126-task run.

## 4. Metrics -- what each one actually measures, and what it doesn't

| metric | measures | does NOT measure |
|---|---|---|
| **hit@3** | did the expected memory appear anywhere in the top-3 results? Binary per task, averaged. | rank quality among the 3, or how wrong a miss was |
| **MRR** | 1/rank if found (1.0 if ranked first, 0.33 if third), 0 if missed. Rank quality on top of hit@3. | — |
| **avg shards contacted** | the router's own cost proxy -- each contacted shard has a real `CostProfile` (latency/$/tokens, see `cost.py`) in the harness's own cost model. Fewer shards contacted = genuinely less spent on the memory-search side of a turn. | the judge's own call cost -- a judge can save shards and still be a net loss if its own call is slow (see §5.3) |
| **judge call latency (added this test)** | wall-clock time of every individual `judge.judge()` call, timed directly (`TimedJudge`), mean/p50/p95 over every call made during the run (a task can trigger more than one judge call if the strategy expands across multiple rounds). | — |
| **exact McNemar p-value (added this test)** | whether a hit@3 difference between two judges *on the same paired tasks* is distinguishable from chance, given how many tasks they actually disagreed on. | effect *size* -- a significant result here says "real," not "large" |

hit@k and MRR are standard information-retrieval metrics, not something
invented for this eval. shards-contacted is specific to this project's
own cost model and is a real, valid cost proxy for the memory-search
side of a turn -- but on its own it's an incomplete efficiency story,
which is exactly what judge latency was added to fix.

## 5. Results

### 5.1 Baseline: does the corpus have a real ceiling now?

All 6 strategies, geometric judge, 209 tasks:

| strategy | hit@3 | MRR | shards contacted |
|---|---|---|---|
| query_all | 0.97 | 0.87 | 3.0/3 |
| flat_topk | 0.97 | 0.87 | 3.0/3 |
| top_n_neighbors | 0.97 | 0.87 | 3.0/3 |
| confidence_threshold | 0.97 | 0.87 | 2.9/3 |
| progressive_expansion | 0.97 | 0.87 | 2.9/3 |
| budgeted_communication | 0.97 | 0.87 | 3.0/3 |

Yes -- even the strategies that search every shard top out at 0.97, not
1.00. A handful of distractor tasks are hard enough that pure vector
similarity occasionally ranks the wrong near-duplicate into the top 3
with no judge involved at all. That's the real ceiling every other
number below is measured against.

### 5.2 Cost and quality: geometric vs. Jev (209 tasks, full corpus)

| strategy | judge | hit@3 | MRR | shards contacted |
|---|---|---|---|---|
| confidence_threshold | geometric | 0.97 | 0.87 | 2.9/3 |
| confidence_threshold | **Jev** | 0.94 | 0.88 | **1.7/3** |
| progressive_expansion | geometric | 0.97 | 0.87 | 2.9/3 |
| progressive_expansion | **Jev** | 0.94 | 0.88 | **1.7/3** |

Jev contacts ~41% fewer shards (1.7 vs. 2.9) for a ~3-point recall cost
(0.97 → 0.94). MRR is actually marginally *higher* for Jev (0.88 vs.
0.87) -- when Jev does find the right memory, it ranks it at least as
well as geometric does; the recall gap is entirely about *whether* it's
found, not how it's ranked once found.

**The 5 tasks Jev missed that geometric got right** (`confidence_threshold`):
- "What budgeting method do I use?" (expected `sem-9`)
- "Did I take the management track offer?" (expected `sem-27`)
- "How do I take my own coffee?" (expected `sem-50`)
- "How do my partner and I settle up shared expenses in practice?" (expected `proc-23`)
- "When did my grandfather pass away?" (expected `sem-56`)

All 5 are cases where Jev decided it had "enough" content before
actually reaching the shard holding the answer -- a real instance of the
failure mode its design implies is possible (stopping early on
confidence, same risk class as `GeometricJudge`'s own stopping logic,
just driven by content-read confidence instead of vector-score gap).
Notably, geometric never lost to Jev the other way on this corpus (0
tasks where Jev found it and geometric didn't) -- the miss pattern is
one-directional, not just noisier in general.

### 5.3 The cost/latency distinction that changes the interpretation

| judge | call latency (p50) | shard savings vs. geometric |
|---|---|---|
| geometric | ~0ms (local math, no network) | — |
| **Jev** | **~750ms** | 1.2 fewer shards (~1.2 × 5ms local shard cost ≈ 6ms) |
| haiku (`claude -p`) | ~4,350–9,070ms | 1.7 fewer shards (126-task run) |

This is the finding that most changes how to read "Jev is more
efficient": **it isn't, in wall-clock terms, for this harness's local
shards.** A local `InMemoryShardStore`/`SQLiteShardStore` shard costs
~5ms to contact (`memory.py`'s own `CostProfile` for episodic/semantic/
procedural). Saving 1.2 of them saves roughly 6ms. Calling Jev to make
that decision costs ~750ms. **The judge call itself dominates total
turn latency by roughly two orders of magnitude over the shard cost it's
optimizing.** What Jev actually saves is downstream **token cost** --
fewer shard results injected into the eventual LLM context window
(`tokens_per_result` in the same `CostProfile`) -- not turn latency.
Whether that trade is worth ~750ms of added judge latency per
judge-sensitive routing decision depends entirely on what the harness is
optimizing for on a given turn, which is a real, harness-level design
question this eval surfaces but does not answer on its own.

(For reference, at 126 tasks the LLM judge, haiku, showed the same
pattern far more severely: 6-12x slower than Jev per call, and a
correspondingly worse recall trade -- 0.85 hit@3 vs. geometric's 0.97,
a difference that *did* reach significance, p=0.0015. Jev also beat
haiku on recall directly and significantly for `confidence_threshold`,
p=0.0075 -- see `results_n126_all_judges.json`.)

### 5.4 Statistical significance: is the recall gap real, or noise?

Exact McNemar test on paired per-task outcomes (same 209 tasks, both
judges) -- only tasks where the two judges *disagree* carry information:

| strategy | geometric-right/Jev-wrong | Jev-right/geometric-wrong | p-value | significant at α=0.05? |
|---|---|---|---|---|
| confidence_threshold | 5 | 0 | **0.0625** | **no -- borderline** |
| progressive_expansion | 5 | 0 | **0.0625** | **no -- borderline** |

At the first pass (126 tasks), this comparison had only 3 discordant
pairs and p=0.25 -- nowhere near conclusive either way. Nearly doubling
the corpus to 209 tasks brought it to 5 discordant pairs and p=0.0625:
right at the conventional significance threshold, one more
geometric-only win away from crossing it (6 discordant, still 0 the
other way, would give p=0.03125).

**This was deliberately not pushed further.** Continuing to add tasks
*specifically until* p drops below 0.05 would be a real methodological
problem -- optional stopping, a well-known way to manufacture
significance that isn't actually there. The honest report at 209 tasks
is: **the recall cost is consistently one-directional (5-0, not
noisier) and close to significant, but not yet proven at a conventional
threshold.** A future run at a larger, pre-committed sample size (not
"add tasks until it's significant") would resolve this properly.

### 5.5 Reliability: does the fallback actually work?

`FallbackJudge` wrapping a `JevJudge` constructed with a deliberately
invalid API key -- a real failed HTTP call every single time, never
mocked -- tested on the 126-task corpus:

| strategy | geometric | jev_fallback | match? |
|---|---|---|---|
| confidence_threshold | hit@3=0.97, shards=2.9 | hit@3=0.97, shards=2.9 | **MATCH** |
| progressive_expansion | hit@3=0.97, shards=2.9 | hit@3=0.97, shards=2.9 | **MATCH** |

Exact match on every strategy, every task. `JevJudge.available` correctly
flips to `False` on the real failure, `FallbackJudge` correctly defers
to `GeometricJudge`, and the harness's behavior under a real Jev outage
is provably identical to never having Jev configured at all -- not
assumed safe, measured safe, on the harder of the two corpora.

## 6. Interpretation

Going back to the three-part hypothesis in §1:

1. **Cost: confirmed.** Jev contacts ~41% fewer shards than geometric,
   consistently, across both corpus sizes tested.
2. **Quality: a real but currently unproven cost.** Jev gives up recall
   one-directionally (never the reverse) on the hardest, most ambiguous
   tasks in this corpus. At 209 tasks the effect is close to
   statistically significant (p=0.0625) but hasn't crossed the
   conventional threshold -- treat it as "probably real, size not yet
   pinned down," not as settled either way.
3. **Reliability: confirmed, with a real (not mocked) failure test.**
   `FallbackJudge` reproduces geometric exactly under an actual Jev
   outage, on every strategy, on the harder corpus.

The efficiency claim needs a caveat this eval surfaced but the original
"20-200x faster" pitch doesn't address on its own: Jev's own call
latency (~750ms) is roughly two orders of magnitude larger than the
local shard cost it's saving (~6ms). It is a genuine token/cost
optimization for this harness's local shards, not a latency
optimization -- the latency win Jev is built for shows up against a
*full LLM call* (haiku took 4.3-9.1 **seconds** per judgment in this
same test), not against a 5ms local SQLite shard.

## 7. Limitations

- All memories are synthetic (LLM-authored, not real user data) --
  plausible and internally consistent, but not sourced from an actual
  Lulu user's session history.
- Ground truth is single-answer by construction (`tasks.py`'s own
  docstring: "kept unambiguous on purpose") -- real user queries can be
  genuinely multi-answer in a way this eval doesn't test.
- 3 shards, 220 memories is still small relative to what a heavy Lulu
  user might accumulate over months (`memory.py`'s own module docstring
  says day-to-day memory "starts small," which is the intended target
  here, not a limitation of the harness itself).
- The geometric-vs-Jev significance result (§5.4) is a borderline,
  underpowered result honestly reported as such -- not a confirmed
  finding either direction.
- Judge latency was measured on this machine, this network path, at
  this moment -- real-world latency will vary with TypeSafe's API load
  and network conditions; treat the ~750ms figure as indicative, not a
  guaranteed SLA.
