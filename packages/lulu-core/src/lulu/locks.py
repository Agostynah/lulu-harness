"""locks.py: advisory, TTL-based file locks to prevent two concurrent
Lulu sessions on the same project from silently racing on the same file.

This is NOT a hard mutex. Nothing ever blocks waiting for a lock, and
there's no acquire/release lifecycle to get wrong on a crash -- a lock is
just a timestamped claim: "session X touched this file recently." Before
write_file/edit_file executes, permissions.py checks for a live claim
from a DIFFERENT session and escalates to ASK if one exists, the same
"no mode skips this" treatment blast_radius.py gets. The tool itself
refreshes the claim right before it writes (see tools/file_tools.py). A
claim simply goes stale after LOCK_TTL_SECONDS -- that's what makes crash
recovery a non-issue: a dead session's lock expires, nothing has to clean
it up, and nothing can get permanently wedged.

This is mutual exclusion, not communication -- it stops two agents from
silently overwriting each other, but says nothing about what either of
them is actually doing. See docs/THESIS.md for why memory (the router),
not this module, is the actual channel for that.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

LOCK_TTL_SECONDS = 300.0  # 5 min: long enough for a slow edit, short
                          # enough that an abandoned session doesn't block
                          # a file indefinitely


@dataclass
class LockClaim:
    path: str
    session_id: str
    timestamp: float

    @property
    def age_seconds(self) -> float:
        return time.time() - self.timestamp


def _lock_file_path(locks_dir: Path, relative_path: str) -> Path:
    # Hash the path rather than mirror its directory structure under
    # locks_dir: sidesteps illegal-filename-character issues entirely and
    # keeps the locks directory flat. The original path is stored inside
    # the lock file's content for debuggability.
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    return locks_dir / f"{digest}.lock"


def check_lock(
    locks_dir: Path, relative_path: str, ttl_seconds: float = LOCK_TTL_SECONDS
) -> LockClaim | None:
    """Returns the live claim on `relative_path`, or None if there isn't
    one -- including if the on-disk claim has aged past `ttl_seconds`, or
    is missing, or is unreadable/corrupt. A corrupt lock file must never
    be treated as a hard block: fail open toward "no lock", not toward
    permanently wedging a file no session can ever touch again.
    """
    lock_path = _lock_file_path(locks_dir, relative_path)
    if not lock_path.exists():
        return None
    try:
        data = json.loads(lock_path.read_text(encoding="utf-8"))
        claim = LockClaim(path=data["path"], session_id=data["session_id"], timestamp=data["timestamp"])
    except (json.JSONDecodeError, KeyError, OSError, TypeError):
        return None
    if claim.age_seconds > ttl_seconds:
        return None
    return claim


def claim_lock(locks_dir: Path, relative_path: str, session_id: str) -> None:
    """Writes (or refreshes) this session's claim on `relative_path`.
    Unconditional -- callers decide via check_lock() whether claiming is
    appropriate; this function just records the claim."""
    locks_dir.mkdir(parents=True, exist_ok=True)
    lock_path = _lock_file_path(locks_dir, relative_path)
    record = {"path": relative_path, "session_id": session_id, "timestamp": time.time()}
    lock_path.write_text(json.dumps(record), encoding="utf-8")
