"""A small, hand-written project memory bank -- decisions, runbooks, and
episodic bug-fix history a real engineering team might actually accumulate.
Every memory here is deliberately plausible and specific, not filler:
tasks.py's ground truth depends on these being distinguishable enough
that "the right answer" is unambiguous to a human reader.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Memory:
    id: str
    shard: str  # "semantic" | "procedural" | "episodic"
    content: str


MEMORIES: list[Memory] = [
    # --- semantic: decisions ---
    Memory(
        "sem-1",
        "semantic",
        "Decided to use PostgreSQL instead of MongoDB for the orders table "
        "because we need transactional consistency across order and inventory writes.",
    ),
    Memory(
        "sem-2",
        "semantic",
        "Decided to rate-limit the public API to 100 requests per minute per API key "
        "to prevent abuse after the incident on 2026-03-14.",
    ),
    Memory(
        "sem-3",
        "semantic",
        "Decided to deprecate the v1 REST API in favor of the v2 GraphQL endpoint; "
        "v1 sunset date is 2026-12-01.",
    ),
    Memory(
        "sem-4",
        "semantic",
        "Decided against using Redis for session storage -- the team lacks ops "
        "experience with it, went with signed cookies instead.",
    ),
    Memory(
        "sem-5",
        "semantic",
        "Decided to split the monolith's billing module into its own service "
        "after it became the top cause of deploy rollbacks.",
    ),
    # --- procedural: runbooks ---
    Memory(
        "proc-1",
        "procedural",
        "Runbook: to rotate the database credentials, first update the secret in "
        "Vault, then restart the connection pool via `kubectl rollout restart "
        "deploy/api`, then verify with `/healthz`.",
    ),
    Memory(
        "proc-2",
        "procedural",
        "Runbook: if the payment webhook queue backs up, check Stripe's status "
        "page first, then drain the dead-letter queue with "
        "`scripts/drain_dlq.py --queue payments`.",
    ),
    Memory(
        "proc-3",
        "procedural",
        "Runbook: to onboard a new engineer, grant GitHub access via the "
        "`engineering` team, add them to `#eng-oncall` on Slack, and assign "
        "the 'first week' checklist in Notion.",
    ),
    Memory(
        "proc-4",
        "procedural",
        "Runbook: when the nightly ETL job fails, check the Airflow DAG "
        "`orders_etl` logs first -- 90% of failures are due to schema drift "
        "in the upstream `orders_raw` table.",
    ),
    Memory(
        "proc-5",
        "procedural",
        "Runbook: to roll back a bad deploy, run `kubectl rollout undo "
        "deploy/api`, then post in `#incidents` with the deploy SHA that was rolled back.",
    ),
    # --- episodic: what actually happened ---
    Memory(
        "epi-1",
        "episodic",
        "Fixed a race condition in the checkout flow where two concurrent "
        "requests could both pass the inventory check and oversell the last "
        "unit -- added a row-level lock on the inventory row.",
    ),
    Memory(
        "epi-2",
        "episodic",
        "Investigated slow p99 latency on /search -- root cause was a missing "
        "index on products.category_id, added it, p99 dropped from 900ms to 60ms.",
    ),
    Memory(
        "epi-3",
        "episodic",
        "Debugged a memory leak in the websocket server -- event listeners on "
        "disconnected sockets were never removed, fixed by calling "
        "`.removeAllListeners()` on disconnect.",
    ),
    Memory(
        "epi-4",
        "episodic",
        "The nightly backup job silently failed for 3 days because the S3 "
        "bucket policy was changed during a security audit -- restored access "
        "and added a Slack alert on backup failure.",
    ),
    Memory(
        "epi-5",
        "episodic",
        "User reported login failures on Safari only -- turned out to be a "
        "SameSite cookie issue, fixed by setting SameSite=None; Secure on the "
        "session cookie.",
    ),
    Memory(
        "epi-6",
        "episodic",
        "Migrated the analytics pipeline from cron-triggered scripts to an "
        "Airflow DAG for better retry and alerting behavior.",
    ),
    Memory(
        "epi-7",
        "episodic",
        "Root-caused a spike in 500 errors to a third-party geocoding API rate "
        "limit -- added exponential backoff and a local cache for repeated addresses.",
    ),
    Memory(
        "epi-8",
        "episodic",
        "Refactored the notification service to use a single outbox table "
        "instead of firing side-effect calls inline, fixing duplicate emails on retry.",
    ),
]
