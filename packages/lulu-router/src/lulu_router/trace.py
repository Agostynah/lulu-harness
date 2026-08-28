"""RoutingTrace: the single object the router produces per query.

One body, three consumers: it renders in the inspector UI (day 5), it's
what gets appended to `.lulu/logs/traces-*.jsonl` (day 4), and it's exactly
what `evals/run.py` aggregates over (day 6). Nothing about routing behavior
is measured any other way -- there is no separate benchmarking
instrumentation that could drift from what a real turn actually does.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass, field

from lulu_router.cost import Budget, Cost
from lulu_router.shard import SearchResult


@dataclass
class ShardScore:
    """One shard's standing in a routing decision, whether or not it was
    ultimately contacted."""

    shard_id: str
    centroid_similarity: float
    contacted: bool
    skip_reason: str | None = None
    # e.g. "outside_top_n", "budget_exhausted", "confidence_met",
    # "sources_exhausted", "denied_by_scope"


@dataclass
class ExpansionRound:
    """One judge verdict during an adaptive-expansion strategy."""

    round_index: int
    shard_id: str
    confidence_before: float
    confidence_after: float
    judge: str
    verdict: str  # "sufficient" | "expand"
    reasoning: str | None = None


@dataclass
class RoutingTrace:
    query: str
    strategy: str
    judge: str
    budget: Budget
    shards_considered: list[ShardScore] = field(default_factory=list)
    rounds: list[ExpansionRound] = field(default_factory=list)
    confidence: float = 0.0
    spent: Cost = field(default_factory=Cost)
    results: list[SearchResult] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)

    @property
    def shards_contacted(self) -> list[str]:
        return [s.shard_id for s in self.shards_considered if s.contacted]

    @property
    def shards_skipped(self) -> list[ShardScore]:
        return [s for s in self.shards_considered if not s.contacted]

    def to_dict(self) -> dict:
        """JSONL-serializable form for `.lulu/logs/traces-*.jsonl`."""
        return dataclasses.asdict(self)
