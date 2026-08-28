"""commands/render.py: formats a RoutingTrace (and its counterfactual)
into the terminal output for /trace and /cost. This is where the thesis
becomes visible every single turn, not just when an eval sweep runs --
see docs/THESIS.md and packages/lulu-core/src/lulu/counterfactual.py.
"""

from __future__ import annotations

from lulu_router.shard import Shard
from lulu_router.trace import RoutingTrace

from lulu.counterfactual import compute_counterfactuals, savings_pct


def render_trace(trace: RoutingTrace) -> str:
    lines = [f"/trace  strategy={trace.strategy}  judge={trace.judge}", "shards:"]
    for s in trace.shards_considered:
        if s.contacted:
            lines.append(f"  ✓ {s.shard_id}  (centroid={s.centroid_similarity:.2f})")
        else:
            lines.append(f"  ✗ {s.shard_id}  ({s.skip_reason}, centroid={s.centroid_similarity:.2f})")

    if trace.rounds:
        lines.append("rounds:")
        for r in trace.rounds:
            lines.append(
                f"  [{r.round_index}] {r.shard_id}: "
                f"{r.confidence_before:.2f} -> {r.confidence_after:.2f} ({r.verdict}) -- {r.reasoning}"
            )

    lines.append(f"confidence: {trace.confidence:.2f}")
    lines.append(
        f"spent: {trace.spent.tokens} tok · {trace.spent.latency_ms:.0f}ms · ${trace.spent.usd:.4f}"
    )
    lines.append(f"results: {len(trace.results)}")
    return "\n".join(lines)


def render_cost(trace: RoutingTrace, shards: list[Shard], k: int) -> str:
    counterfactuals = compute_counterfactuals(shards, k)

    lines = [f"/cost  strategy={trace.strategy}  judge={trace.judge}"]

    shard_bits = [
        f"{s.shard_id} ✓" if s.contacted else f"{s.shard_id} ✗({s.skip_reason})"
        for s in trace.shards_considered
    ]
    lines.append("shards: " + "  ".join(shard_bits))

    lines.append(
        f"spent: {trace.spent.tokens} tok · {trace.spent.latency_ms:.0f}ms · ${trace.spent.usd:.4f}"
    )

    for cf in counterfactuals:
        pct = savings_pct(trace.spent, cf.cost, attr="tokens")
        lines.append(
            f"{cf.label} would have been: {cf.cost.tokens} tok · {cf.cost.latency_ms:.0f}ms · "
            f"${cf.cost.usd:.4f}   -> {pct:+.0f}% tok"
        )

    return "\n".join(lines)
