"""GeometricJudge: the paper's confidence estimate, sigmoid(gap) x coverage.

Ported from local-memory/core/confidence.py (distributed-vector-memory-
routing), but with the gap/steepness recalibrated to the *notebook's*
validated formula rather than local-memory's, which turned out to differ
in a way that matters:

  - local-memory: gap = top1 - second_score, steepness = 1.0
  - paper notebook: gap = top1 - mean(rest of top-k), steepness = 5.0

local-memory's version was never actually validated on a real corpus (see
the "6 hand-tuned signals" note in this project's plan) -- and it shows:
on the real DBpedia14 sweep (evals/dbpedia), steepness=1.0 produced a
sigmoid too flat to clear any reasonable threshold even at full coverage,
so confidence_threshold/progressive_expansion degenerated into always
contacting every shard, identical to query_all. Switching to the paper's
own validated parameters (steepness=5.0, gap against the mean of the rest
of top-k, threshold=0.5) restored early stopping. This is the sort of
thing that's supposed to get caught by evals/dbpedia and reported, not
buried -- see docs/THESIS.md's falsification discipline.

This judge reads nothing about the content of what came back -- it's
purely a function of score separation and how much of the shard space has
been searched. It's the judges/ package's baseline: ClaudeCLIJudge is
measured against this one for both quality and cost, not assumed to beat
it (see docs/THESIS.md, Contribution #2).
"""

from __future__ import annotations

import math

from lulu_router.shard import SearchResult

DEFAULT_THRESHOLD = 0.5
DEFAULT_STEEPNESS = 5.0


def _sigmoid(x: float, steepness: float) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * x))


class GeometricJudge:
    name = "geometric"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, steepness: float = DEFAULT_STEEPNESS):
        self.threshold = threshold
        self.steepness = steepness

    def judge(
        self,
        query: str,
        results: list[SearchResult],
        sources_contacted: int,
        total_sources: int,
    ) -> tuple[float, bool, str]:
        if not results:
            return 0.0, True, "no results yet"
        top_score = results[0].score
        rest_scores = [r.score for r in results[1:]]
        rest_mean = sum(rest_scores) / len(rest_scores) if rest_scores else 0.0
        gap = top_score - rest_mean
        coverage = sources_contacted / max(total_sources, 1)
        confidence = _sigmoid(gap, self.steepness) * coverage
        should_expand = confidence < self.threshold
        reasoning = f"gap={gap:.3f} coverage={coverage:.2f} -> confidence={confidence:.3f}"
        return confidence, should_expand, reasoning
