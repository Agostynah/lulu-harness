"""permissions.py: the harness's attention-interface enforcement point.

Every side-effecting tool call goes through PermissionChecker.check()
before loop.py ever calls registry.dispatch(). Decision precedence, in
strict order:

1. Read-only tools (read_file, glob, grep) -- always ALLOW. They have no
   blast radius, and asking on every read would make the harness unusable.
2. A blast-radius match on a bash command -- always ASK, regardless of
   mode. Not even AUTO mode skips this (see blast_radius.py).
3. The active AttentionMode's preset for this tool.
4. Fail-safe default: ASK, for any tool the preset table doesn't know
   about -- adding a new side-effecting tool without updating
   attention.MODE_PRESETS is safe by construction, not silently permissive.

Every decision is logged to .lulu/logs/permissions.jsonl, including the
*actual* human outcome for an ASK once loop.py has one -- that log is what
attention.suggest_promotions() later aggregates, and what /cost and /trace
also read from (one log, several consumers, same pattern as RoutingTrace).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lulu.attention import MODE_PRESETS, AttentionMode, Decision
from lulu.blast_radius import assess_blast_radius
from lulu.locks import check_lock

READ_ONLY_TOOLS = frozenset({"read_file", "glob", "grep"})
LOCKABLE_TOOLS = frozenset({"write_file", "edit_file"})
MAX_LOGGED_ARG_CHARS = 500


def _summarize_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Truncates long string arguments (e.g. write_file's `content`)
    before logging -- the permissions log is meant to stay grep-able and
    small, same philosophy as the read_file size cap and bash output
    truncation elsewhere in this package."""
    summarized: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and len(value) > MAX_LOGGED_ARG_CHARS:
            omitted = len(value) - MAX_LOGGED_ARG_CHARS
            summarized[key] = value[:MAX_LOGGED_ARG_CHARS] + f"...[{omitted} more chars]"
        else:
            summarized[key] = value
    return summarized


@dataclass
class PermissionResult:
    decision: Decision
    reason: str
    blast_radius_reasons: list[str]


class PermissionChecker:
    def __init__(
        self,
        mode: AttentionMode = AttentionMode.MANUAL,
        log_path: Path | None = None,
        locks_dir: Path | None = None,
        session_id: str | None = None,
    ):
        self.mode = mode
        self.log_path = log_path
        self.locks_dir = locks_dir
        self.session_id = session_id

    def check(self, tool_name: str, arguments: dict[str, Any]) -> PermissionResult:
        if tool_name in READ_ONLY_TOOLS:
            return PermissionResult(Decision.ALLOW, "read-only tool", [])

        if tool_name == "bash":
            reasons = assess_blast_radius(arguments.get("command", ""))
            if reasons:
                return PermissionResult(Decision.ASK, "blast-radius: " + "; ".join(reasons), reasons)

        if tool_name in LOCKABLE_TOOLS and self.locks_dir is not None and "path" in arguments:
            claim = check_lock(self.locks_dir, arguments["path"])
            if claim is not None and claim.session_id != self.session_id:
                reason = (
                    f"file locked by another session ({claim.session_id}, "
                    f"{claim.age_seconds:.0f}s ago) -- possible concurrent edit"
                )
                return PermissionResult(Decision.ASK, reason, [])

        preset = MODE_PRESETS.get(self.mode, {})
        if tool_name in preset:
            return PermissionResult(preset[tool_name], f"mode={self.mode.value} preset", [])

        return PermissionResult(Decision.ASK, "no preset for this tool (fail-safe default)", [])

    def log(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        result: PermissionResult,
        outcome: str,
    ) -> None:
        """outcome is what actually happened -- 'auto_allowed',
        'auto_denied', 'approved', or 'denied' -- which for an ASK
        decision is determined by loop.py after a human responds, and may
        of course differ from result.decision itself (that's the whole
        point of ASK)."""
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": time.time(),
            "tool": tool_name,
            "arguments": _summarize_arguments(arguments),
            "mode": self.mode.value,
            "decision": result.decision.value,
            "reason": result.reason,
            "blast_radius_reasons": result.blast_radius_reasons,
            "outcome": outcome,
        }
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
