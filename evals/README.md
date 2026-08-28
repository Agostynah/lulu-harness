# Evals

Three evals, three different questions. Each one reads the exact
`RoutingTrace`-producing code paths the harness uses at runtime, not a
separate benchmark instrumentation — see `docs/THESIS.md` for why that
matters.

- **`dbpedia/`** — replicates the
  [distributed-vector-memory-routing](https://github.com/Agostynah/distributed-vector-memory-routing)
  paper's 100K-document experiment against `lulu_router`'s own strategy
  code, plus the axis the paper didn't have (geometric vs. LLM judge). See
  `dbpedia/README.md`.
- **`agent_tasks/`** — the geometric-vs-LLM-judge comparison DBpedia14
  can't fairly give you, because its "queries" are similarity probes, not
  real questions. 25 hand-labeled tasks against an 18-memory project bank.
  See `agent_tasks/README.md`.
- **`leakage.py`** (this file) — the permission-boundary demo. Binary,
  no interpretation needed.

## `leakage.py`: the permission-boundary demo

The core claim of `docs/THESIS.md`'s Contribution #1 is that a flat,
merged vector index *cannot* express a permission boundary, not just that
it's more expensive to add one. This script makes that concrete instead
of asserted: it builds two customers' memories with realistically
overlapping topics (`SEPARATION=0.5`, empirically swept so the clusters
actually overlap in vector space — see the script's docstring for why a
larger separation would prove nothing), then runs the same 200 queries
from a `customer-a`-scoped agent against two things —

- **Lulu, routed and scoped**: `MemoryRouter` checks `Shard.permits()`
  before any strategy runs, so a `customer-b` shard is invisible to a
  `customer-a` scope regardless of which of the 6 strategies is asked.
- **A genuinely flat index**: every vector merged into one
  `InMemoryShardStore` with no record of which customer it came from —
  there is no shard boundary left to attach a permission check to, so
  there's nothing to scope the search by at all.

```bash
uv run python evals/leakage.py
uv run python evals/leakage.py --strategy progressive_expansion
uv run python evals/leakage.py --n-queries 500 --separation 0.3   # harder test
```

### Results (200 queries, k=5, separation=0.5)

| | leaked queries | leak rate | customer-b memories surfaced |
|---|---|---|---|
| Lulu (scoped routing) | 0/200 | 0.0% | 0 |
| flat index (no shard boundary) | 53/200 | 26.5% | 98 |

Confirmed across all 6 strategies (`--strategy query_all\|flat_topk\|top_n_neighbors\|confidence_threshold\|progressive_expansion\|budgeted_communication`)
— every one of them checks scope at the same point, before strategy
dispatch, so which strategy runs never changes the result. That includes
`flat_topk`: even lulu_router's own "merge everything and rank" strategy
still respects the permission check, because the check happens at
shard-ranking time, not inside the strategy. The genuinely flat index
built by this script has no such check available to it at all — that's
the actual mechanism gap, not a configuration choice.

"Why not a flat index, it's simpler" stops being answerable with "it's
cheaper" and becomes "the flat one *can't* — not won't, can't."
