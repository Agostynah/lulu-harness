# Thesis

Source paper: [distributed-vector-memory-routing](https://github.com/Agostynah/distributed-vector-memory-routing)
(DBpedia14, 100K docs, 16 KMeans partitions, 4 adaptive strategies —
**42% less communication at 100% recall @ K=16**).

## The reformulation

A distributed system pays for network hops. A single-machine agent harness
doesn't — so "why shard at all, a flat index is simpler" is the obvious,
correct-sounding objection. It's wrong, but only once you name the actual
scarce resource.

**The scarce resource in a harness is not network bytes. It's context
tokens, latency, and dollars.** Every memory shard has a cost to *inject*
one of its results into the model's context window, and that cost is not
uniform:

| shard | latency | $/query | tokens/result |
|---|---|---|---|
| local SQLite/TurboVec (episodic, semantic) | ~5ms | 0 | ~80–150 |
| local code index | ~20ms | 0 | ~600 |
| MCP connector (remote service) | ~800ms | real | ~300 |

A flat top-k index over one merged corpus **cannot express** "I already have
high confidence from local shards, don't pay the remote shard's latency and
$." Once vectors are merged, every result costs the same to fetch, by
construction — there's nowhere to attach a cost distinction. A router that
tracks cost per shard and spends against an explicit `Budget` can express
exactly that. That's what turns `progressive_expansion` and
`budgeted_communication` from paper theory into a real cost controller.

**Falsifiability, stated up front:** if `flat_topk` over a single merged
index matches routed recall at equal-or-lower token cost — *with the
`remote` shard active in the mix* — the sharding is decorative and the
design degrades to just the budget controller. That result gets reported
either way; see `evals/run.py`'s success criterion.

## Contribution #1 — the shard as a unit of cost, volatility, *and permission*

The paper partitions purely by semantics (KMeans). Inside a harness, two
more axes fall out for free, and the permission axis is the one that makes
sharding irreplaceable rather than merely efficient:

```
shards_to_contact = permitted(caller_identity) ∩ worth_it(centroid, cost, budget, freshness)
```

Things a flat index cannot do **even in principle**, because the vectors
are already mixed with nowhere to put a boundary:

- A subagent scoped to *customer A* can never retrieve *customer B*'s
  memories.
- A `local-only` shard that is never routed into a prompt headed to a
  third-party model.
- A `quarantine` shard: memories written from untrusted tool output are
  stored but never enter a privileged context — prompt-injection
  containment via the same mechanism.
- Launching a subagent hands it a **routing policy**, not a memory dump.
  Write permissions for tools *and* read permissions for memory — almost no
  harness does the second one.

**The demo that ends the argument, binary, no interpretation needed:** 200
queries from an agent scoped to customer A. `flat_topk` leaks customer B's
memories (fails). Routing with a policy → 0 leaks, same recall within A.
"Why not a flat index" stops being answered with "it's cheaper" and becomes
"the flat one *can't*." (`evals/leakage.py`, day 6.)

**Volatility** (cheaper, practical): memory decays at different rates — a
code index invalidates on every commit, episodic memory almost never does.
Partitioning by write-rate too means only the hot shard needs
re-indexing, and the router can discount or refresh a shard it knows is
stale. A flat index re-embeds everything or serves stale results.

## Contribution #2 — a learned stopping criterion (the small-model judge)

The paper's confidence estimate is `sigmoid(gap) × coverage` — a purely
geometric proxy that never reads the actual content of what came back
(`judges/geometric.py`, ported from `local-memory/core/confidence.py`).

v2 adds a second judge that *does* read the content: after each expansion
round, a small model looks at the accumulated candidates and decides "is
this enough for the task, and if not, what's missing?" v0 shells out to
`claude -p --model haiku` (see roadmap for local/API backends). Both
judges satisfy the same `Judge` protocol in `strategies.py`, so any
strategy can run under either one interchangeably.

That makes the judge itself an evaluable axis: **geometric stopping vs.
LLM-judge stopping vs. fixed-k**, each scored on retrieval quality against
its own cost. If the judge runs fully local (Ollama), the entire retrieval
path costs $0 and never leaves the machine — only the harness's main loop
calls a frontier API. That property is what the `local-only` shard in
Contribution #1 is for.

## Falsification discipline

Every claim above is checked against a named baseline in `evals/run.py`,
not asserted:

- **Sharding at all** → beaten if `flat_topk` matches routed recall at
  equal/lower cost with `remote` active.
- **Adaptive expansion over fixed k** → beaten if `top_n_neighbors` (fixed
  N, no judge) matches `progressive_expansion`/`confidence_threshold` at
  equal cost.
- **LLM judge over geometric judge** → reported as a quality/cost
  trade-off, not assumed to win; `evals/dbpedia` runs both.
- **Permission boundary claim** → binary pass/fail via `evals/leakage.py`,
  not a recall number at all.
