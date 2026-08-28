"""MemoryRouter: the single entry point strategies.py's implementations are
dispatched through. This is what the harness's ContextAssembler (day 4)
calls once per turn to decide what goes into the context window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lulu_router.cost import Budget
from lulu_router.shard import Shard
from lulu_router.strategies import STRATEGIES, Judge
from lulu_router.trace import RoutingTrace


@dataclass
class MemoryRouter:
    shards: list[Shard]
    judge: Judge

    def route(
        self,
        query: str,
        query_vec: np.ndarray,
        strategy: str = "progressive_expansion",
        budget: Budget | None = None,
        k: int = 10,
        scope: str | None = None,
    ) -> RoutingTrace:
        if strategy not in STRATEGIES:
            raise ValueError(f"unknown strategy: {strategy!r}. options: {sorted(STRATEGIES)}")
        fn = STRATEGIES[strategy]
        return fn(
            query=query,
            query_vec=query_vec,
            shards=self.shards,
            budget=budget or Budget(),
            k=k,
            judge=self.judge,
            scope=scope,
        )
