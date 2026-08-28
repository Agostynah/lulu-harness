# DBpedia14 sweep

Reproduces the [distributed-vector-memory-routing](https://github.com/Agostynah/distributed-vector-memory-routing)
paper's experiment against `lulu_router`'s own strategy implementations,
plus the axis the paper didn't have: geometric confidence vs. an LLM-judge
stopping criterion. See `run.py`'s module docstring for the two documented
methodology deviations (corpus size, embedding model).

```bash
# fast dev run (default: 3000 docs, K in {2,4,8}, 50 queries)
uv run python evals/dbpedia/run.py

# original paper's regime (100K docs, K in {2,4,8,16}) -- slow, needs a while
uv run python evals/dbpedia/run.py --n-docs 100000 --partition-counts 2,4,8,16

# add the LLM-judge axis (shells out to `claude -p --model haiku` per judgment,
# so kept to a handful of queries by default)
uv run python evals/dbpedia/run.py --llm-judge --llm-judge-queries 5
```

Embeddings are cached to `.cache/` (git-ignored) keyed by corpus size and
model, so re-running with `--llm-judge` after an initial run doesn't
re-embed.

## First real numbers (3000 docs, K=2/4/8/16, 50 queries, geometric judge)

| K | strategy | Recall@10 | shards contacted | comm saved |
|---|---|---|---|---|
| 4 | top_n_neighbors | 1.00 | 3/4 | 25% |
| 8 | confidence_threshold / progressive_expansion | 0.99 | 5/8 | 37% |
| 8 | budgeted_communication (cap = K/2) | 0.98 | 4/8 | 50% |
| 16 | confidence_threshold / progressive_expansion | 0.99 | 10/16 | 38% |
| 16 | budgeted_communication (cap = K/2) | 0.98 | 8/16 | 50% |
| 16 | top_n_neighbors (n=3) | 0.85 | 3/16 | 81% |

`flat_topk` matches `query_all` exactly at every K (expected: a merged
index can't decline a shard). The strict "beats flat_topk" bar (recall ≥
AND cost <) only clears at K=4; at K=8/16 routing trades ~1 point of
recall for 37-50% fewer shards contacted -- a genuine trade-off, not a
free lunch, and reported as such rather than rounded up.

**Note on the geometric judge's calibration**: the first version of this
sweep used local-memory's ported `confidence.py` parameters
(`steepness=1.0`, `gap = top1 - second_score`) and found
`confidence_threshold`/`progressive_expansion` never stopped early at
*any* K -- confidence never crossed the threshold even at full coverage,
so they silently degenerated into `query_all`. Recalibrating to the
paper notebook's own validated parameters (`steepness=5.0`,
`gap = top1 - mean(rest of top-k)`, `threshold=0.5`) fixed it; see
`judges/geometric.py`'s docstring. Caught by this eval, not assumed away
-- exactly the falsification discipline docs/THESIS.md commits to.

**LLM-judge plumbing check** (K=8, 2 queries, real document text as query
and candidate content): `ClaudeCLIJudge` wires correctly into both
judge-sensitive strategies and drives materially different routing
decisions than the geometric judge (stops after ~1 shard instead of 5),
but at a real recall cost (0.45-0.65 vs. 0.99) on this benchmark. As
documented in `run.py`'s docstring, DBpedia14 has no real task queries for
a sufficiency judge to reason about -- this confirms the wiring works, not
the judge's quality. That comparison is evals/agent_tasks's job (day 6).
