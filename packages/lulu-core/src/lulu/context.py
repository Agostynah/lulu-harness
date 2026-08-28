"""context.py: ContextAssembler -- the integration seam between the two
halves of this project.

packages/lulu-router has no idea a harness exists; it's tested and
evaluated (evals/dbpedia) entirely on its own. This module is the only
place in lulu-core that imports it: ContextAssembler calls
MemoryRouter.route() and renders whatever RoutingTrace comes back into an
injectable text block. Nothing about routing strategies, judges, or shard
partitioning leaks past this file into the rest of the harness.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lulu_router.cost import Budget
from lulu_router.router import MemoryRouter
from lulu_router.trace import RoutingTrace

DEFAULT_STRATEGY = "progressive_expansion"
DEFAULT_K = 10


@dataclass
class AssembledContext:
    text: str
    trace: RoutingTrace


class ContextAssembler:
    def __init__(
        self,
        router: MemoryRouter,
        strategy: str = DEFAULT_STRATEGY,
        k: int = DEFAULT_K,
    ) -> None:
        self.router = router
        self.strategy = strategy
        self.k = k

    def assemble(
        self,
        query: str,
        query_vec: np.ndarray,
        budget: Budget,
        scope: str | None = None,
    ) -> AssembledContext:
        trace = self.router.route(
            query, query_vec, strategy=self.strategy, budget=budget, k=self.k, scope=scope
        )
        return AssembledContext(text=self._render(trace), trace=trace)

    @staticmethod
    def _render(trace: RoutingTrace) -> str:
        if not trace.results:
            return ""
        lines = ["# Relevant memory", ""]
        lines.extend(f"- ({r.score:.2f}) {r.content}" for r in trace.results)
        return "\n".join(lines)
