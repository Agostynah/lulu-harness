"""lulu-router: adaptive routing across cost-heterogeneous memory shards.

The standalone core of the Lulu project's thesis (see ../../../docs/THESIS.md):
a query is routed to some subset of shards -- not all of them -- based on
semantic proximity (centroid similarity), an explicit cost budget (tokens /
latency / $), an access policy (which shards this caller may even see), and
a Judge's assessment of when accumulated results are "enough". This package
has no dependency on the harness: it's tested and benchmarked (see
evals/dbpedia) entirely on its own.
"""

from lulu_router.cost import Budget, Cost, CostProfile
from lulu_router.judges import ClaudeCLIJudge, GeometricJudge
from lulu_router.router import MemoryRouter
from lulu_router.shard import SearchResult, Shard, ShardStore
from lulu_router.strategies import STRATEGIES, Judge
from lulu_router.trace import ExpansionRound, RoutingTrace, ShardScore

__all__ = [
    "Budget",
    "Cost",
    "CostProfile",
    "ClaudeCLIJudge",
    "GeometricJudge",
    "MemoryRouter",
    "SearchResult",
    "Shard",
    "ShardStore",
    "STRATEGIES",
    "Judge",
    "ExpansionRound",
    "RoutingTrace",
    "ShardScore",
]
