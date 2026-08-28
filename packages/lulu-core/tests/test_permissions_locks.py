"""PermissionChecker's lock-conflict escalation: another session's live
claim on the same file forces ASK regardless of mode (same treatment as
blast-radius), your own session's claim never counts as a conflict, and
locking is a no-op entirely when locks_dir isn't configured (backward
compatible with every PermissionChecker usage that predates this
feature)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lulu.attention import AttentionMode, Decision
from lulu.locks import claim_lock
from lulu.permissions import PermissionChecker


def test_no_conflict_when_locks_dir_not_configured(tmp_path: Path):
    checker = PermissionChecker(mode=AttentionMode.AUTO, session_id="session-a")
    result = checker.check("write_file", {"path": "a.py", "content": "x"})
    assert result.decision == Decision.ALLOW  # falls through to the AUTO preset, no lock check at all


def test_no_conflict_when_no_lock_exists(tmp_path: Path):
    checker = PermissionChecker(mode=AttentionMode.AUTO, locks_dir=tmp_path, session_id="session-a")
    result = checker.check("write_file", {"path": "a.py"})
    assert result.decision == Decision.ALLOW


@pytest.mark.parametrize("mode", list(AttentionMode))
def test_other_sessions_live_claim_forces_ask_regardless_of_mode(tmp_path: Path, mode: AttentionMode):
    """Same precedence already established for blast-radius
    (test_permissions.py::test_plan_mode_denies_non_blast_radius_bash_but_blast_radius_still_asks):
    an escalation to ASK is never less safe than a blanket DENY -- nothing
    executes without a human saying yes either way -- so it's allowed to
    override PLAN's own deny-everything preset too, consistently."""
    claim_lock(tmp_path, "a.py", session_id="session-other")
    checker = PermissionChecker(mode=mode, locks_dir=tmp_path, session_id="session-mine")

    result = checker.check("write_file", {"path": "a.py"})

    assert result.decision == Decision.ASK
    assert "session-other" in result.reason


def test_own_sessions_claim_is_not_a_conflict(tmp_path: Path):
    claim_lock(tmp_path, "a.py", session_id="session-mine")
    checker = PermissionChecker(mode=AttentionMode.AUTO, locks_dir=tmp_path, session_id="session-mine")

    result = checker.check("write_file", {"path": "a.py"})

    assert result.decision == Decision.ALLOW


def test_expired_claim_from_another_session_is_not_a_conflict(tmp_path: Path):
    import json
    import time

    from lulu.locks import LOCK_TTL_SECONDS

    claim_lock(tmp_path, "a.py", session_id="session-other")
    lock_files = list(tmp_path.glob("*.lock"))
    record = json.loads(lock_files[0].read_text(encoding="utf-8"))
    record["timestamp"] = time.time() - (LOCK_TTL_SECONDS + 1)
    lock_files[0].write_text(json.dumps(record), encoding="utf-8")

    checker = PermissionChecker(mode=AttentionMode.AUTO, locks_dir=tmp_path, session_id="session-mine")
    result = checker.check("write_file", {"path": "a.py"})

    assert result.decision == Decision.ALLOW


def test_read_only_tools_never_check_locks(tmp_path: Path):
    """Reading a locked file is fine -- only concurrent writes race."""
    claim_lock(tmp_path, "a.py", session_id="session-other")
    checker = PermissionChecker(mode=AttentionMode.AUTO, locks_dir=tmp_path, session_id="session-mine")

    result = checker.check("read_file", {"path": "a.py"})

    assert result.decision == Decision.ALLOW


def test_lock_conflict_on_a_different_path_does_not_interfere(tmp_path: Path):
    claim_lock(tmp_path, "other.py", session_id="session-other")
    checker = PermissionChecker(mode=AttentionMode.AUTO, locks_dir=tmp_path, session_id="session-mine")

    result = checker.check("write_file", {"path": "a.py"})

    assert result.decision == Decision.ALLOW
