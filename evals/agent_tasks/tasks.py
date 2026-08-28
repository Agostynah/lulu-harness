"""~25 hand-labeled tasks against memories.py's memory bank. Real
natural-language questions a developer would actually ask, not similarity
probes -- this is what makes this eval a fair test of the LLM judge,
unlike evals/dbpedia (see its README for why). About a third of these are
deliberate paraphrases of the target memory's own wording, specifically
to test semantic retrieval rather than keyword overlap.

Each task has exactly one expected memory id -- kept unambiguous
on purpose so ground truth stays defensible by inspection, not a judgment
call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    query: str
    expected_id: str


TASKS: list[Task] = [
    Task("Why did we choose Postgres over Mongo for orders?", "sem-1"),
    Task("What's the API rate limit and why was it added?", "sem-2"),
    Task("When is the old REST API going away?", "sem-3"),
    Task("Why aren't we using Redis for sessions?", "sem-4"),
    Task("Why did billing get split out of the monolith?", "sem-5"),
    Task("How do I rotate database credentials?", "proc-1"),
    Task("The payment webhook queue is backed up, what do I do?", "proc-2"),
    Task("What's the process for onboarding a new engineer?", "proc-3"),
    Task("The nightly ETL job failed, where do I look first?", "proc-4"),
    Task("How do I roll back a bad deploy?", "proc-5"),
    Task("Have we seen overselling bugs in checkout before?", "epi-1"),
    Task("Why was /search slow before, and how was it fixed?", "epi-2"),
    Task("Was there ever a memory leak in the websocket server?", "epi-3"),
    Task("Has our backup job ever failed silently?", "epi-4"),
    Task("Any past issues with login on Safari?", "epi-5"),
    Task("Did we ever move analytics off of cron jobs?", "epi-6"),
    Task("Have we had rate limit issues with the geocoding API?", "epi-7"),
    Task("Was there a duplicate email bug before?", "epi-8"),
    # paraphrases -- same expected answer as an item above, different wording
    Task("If I need to restart the API pods after a secret rotation, what's the runbook?", "proc-1"),
    Task("What should I check first when the Airflow orders_etl DAG fails overnight?", "proc-4"),
    Task("Give me the steps to undo a broken production release.", "proc-5"),
    Task("Why don't we store sessions in Redis?", "sem-4"),
    Task("What caused the checkout system to oversell inventory, and how was it addressed?", "epi-1"),
    Task("How was the p99 latency issue on search resolved?", "epi-2"),
    Task("What's our plan for sunsetting API v1?", "sem-3"),
]
