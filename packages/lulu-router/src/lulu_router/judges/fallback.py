"""FallbackJudge: tries `primary` first, falls back to `secondary` only
when `primary` genuinely couldn't reach a verdict -- not when it reached
one and it happened to be low-confidence, which is a real answer, not a
failure.

Built for the Jev-first, geometric-as-fallback default (see
lulu.memory.default_judge): Jev is faster and cheaper and reads content,
but it's an external API call that can be unreachable (no key, network
down, TypeSafe outage) in a way GeometricJudge -- pure local math, no I/O
-- structurally cannot be. Wrapping them keeps the router's uptime tied to
local math even when the network isn't, without giving up Jev's better
answer the rest of the time.

Generic over any `primary` that opts in by setting a public `available`
bool after each `judge()` call (see JevJudge) -- a primary that never sets
it is just always trusted, same as not wrapping it at all.
"""

from __future__ import annotations

from lulu_router.shard import SearchResult


class FallbackJudge:
    def __init__(self, primary, secondary) -> None:
        self.primary = primary
        self.secondary = secondary
        self.name = f"{primary.name}+{secondary.name}"

    def judge(
        self,
        query: str,
        results: list[SearchResult],
        sources_contacted: int,
        total_sources: int,
    ) -> tuple[float, bool, str]:
        confidence, should_expand, reasoning = self.primary.judge(query, results, sources_contacted, total_sources)
        if getattr(self.primary, "available", True):
            return confidence, should_expand, reasoning

        fb_confidence, fb_should_expand, fb_reasoning = self.secondary.judge(
            query, results, sources_contacted, total_sources
        )
        return fb_confidence, fb_should_expand, f"{reasoning} -> fell back: {fb_reasoning}"
