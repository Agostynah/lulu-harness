"""AttentionMode: the harness's attention-interface pilot.

Dan McAteer's "The Evolution of the Agent Harness" (2026) argues that as
models absorb harness capabilities, human attention is the one resource
that can't be absorbed away -- the harness ends up being the interface
*to* the human, not to the machine. The expensive version of that idea is
a learned attention policy. This is the cheap version: a named, switchable
preset over permissions.py (the same shape as Claude Code's own
manual/plan/accept-edits/bypass modes), plus a non-ML suggestion mechanism
that just counts approvals in the log that already gets written.

Nothing here can grant itself more autonomy. suggest_promotions() only
ever proposes -- applying a suggestion is a separate, explicit human
action. See docs/THESIS.md, Contribution #3.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


class AttentionMode(str, Enum):
    MANUAL = "manual"
    PLAN = "plan"
    AUTO_EDITS = "auto_edits"
    AUTO = "auto"


class Decision(str, Enum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


# Presets for side-effecting tools only -- read-only tools (read_file,
# glob, grep) are always ALLOW regardless of mode; see permissions.py.
# Unlisted tools fall back to ASK (fail-safe default), so adding a new
# side-effecting tool without updating this table is safe by construction.
MODE_PRESETS: dict[AttentionMode, dict[str, Decision]] = {
    AttentionMode.MANUAL: {
        "write_file": Decision.ASK,
        "edit_file": Decision.ASK,
        "bash": Decision.ASK,
    },
    AttentionMode.PLAN: {
        "write_file": Decision.DENY,
        "edit_file": Decision.DENY,
        "bash": Decision.DENY,
    },
    AttentionMode.AUTO_EDITS: {
        "write_file": Decision.ALLOW,
        "edit_file": Decision.ALLOW,
        "bash": Decision.ASK,
    },
    AttentionMode.AUTO: {
        "write_file": Decision.ALLOW,
        "edit_file": Decision.ALLOW,
        "bash": Decision.ALLOW,
    },
}


@dataclass
class PromotionSuggestion:
    tool: str
    current_decision: Decision
    approval_streak: int
    reason: str


def suggest_promotions(
    log_path: Path,
    min_streak: int = 5,
    mode: AttentionMode = AttentionMode.MANUAL,
) -> list[PromotionSuggestion]:
    """Reads permissions.jsonl and proposes ask -> allow promotions for
    tools that have been approved `min_streak` times in a row with zero
    denials since. Pure aggregation over the existing log -- no ML, no
    separate state, nothing that wasn't already being written for
    /cost and /trace. Never applies anything; the caller (the /mode
    suggest command) decides whether to act on a suggestion.

    A denial anywhere in a tool's history resets its streak counter --
    one "no" outweighs any number of prior "yes"es, since the log is
    read oldest-first and streaks are counted from the most recent
    denial (or the start of the log) forward.
    """
    if not log_path.exists():
        return []

    current_preset = MODE_PRESETS.get(mode, {})
    streaks: dict[str, int] = defaultdict(int)

    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            tool = record.get("tool")
            outcome = record.get("outcome")
            if tool is None or outcome is None:
                continue
            if outcome == "approved":
                streaks[tool] += 1
            elif outcome == "denied":
                streaks[tool] = 0

    suggestions: list[PromotionSuggestion] = []
    for tool, streak in streaks.items():
        if streak < min_streak:
            continue
        current = current_preset.get(tool, Decision.ASK)
        if current == Decision.ALLOW:
            continue  # already allowed under this mode, nothing to suggest
        suggestions.append(
            PromotionSuggestion(
                tool=tool,
                current_decision=current,
                approval_streak=streak,
                reason=f"approved {streak} times in a row with no denials since",
            )
        )
    return suggestions
