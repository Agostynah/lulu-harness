"""Cost accounting for memory shards.

The central bet of this router: shards are not interchangeable. A local
SQLite/TurboVec shard costs ~0 to query; a shard behind an MCP connector to
a remote service costs real latency and real dollars. A flat top-k index
over a single merged corpus cannot express this distinction, because once
everything is merged into one index, every result costs the same to fetch.
The router can express it, because it tracks cost per shard and spends
against an explicit budget. See ../../../docs/THESIS.md.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CostProfile:
    """Per-query cost of contacting one shard, measured (preferred) or
    estimated. `tokens_per_result` is what it costs to inject one result
    from this shard into the model's context window -- that's the number
    that actually matters for a harness, more than raw retrieval latency."""

    latency_ms: float
    usd_per_query: float
    tokens_per_result: int

    def cost_for(self, n_results: int) -> Cost:
        return Cost(
            latency_ms=self.latency_ms,
            usd=self.usd_per_query,
            tokens=self.tokens_per_result * n_results,
        )


@dataclass
class Cost:
    """Accumulated cost of a routing decision, real or counterfactual."""

    latency_ms: float = 0.0
    usd: float = 0.0
    tokens: int = 0

    def __add__(self, other: Cost) -> Cost:
        return Cost(
            latency_ms=self.latency_ms + other.latency_ms,
            usd=self.usd + other.usd,
            tokens=self.tokens + other.tokens,
        )

    def exceeds(self, budget: Budget) -> bool:
        return (
            self.tokens > budget.max_tokens
            or self.latency_ms > budget.max_latency_ms
            or self.usd > budget.max_usd
        )


@dataclass(frozen=True)
class Budget:
    """A hard cap the router spends against. Defaults are generous; callers
    (the harness's ContextAssembler, or an eval sweep) are expected to pass
    a real one."""

    max_tokens: int = 2000
    max_latency_ms: float = 500.0
    max_usd: float = 0.01
