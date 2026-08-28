"""locks.py: advisory TTL-based claims. Adversarial focus: a corrupt or
partial lock file must fail OPEN (no lock), never wedge a file shut, and
an expired claim must be treated as absent."""

from __future__ import annotations

import json
import time
from pathlib import Path

from lulu.locks import LOCK_TTL_SECONDS, LockClaim, check_lock, claim_lock


def test_no_lock_file_means_no_claim(tmp_path: Path):
    assert check_lock(tmp_path, "src/main.py") is None


def test_claim_then_check_returns_it(tmp_path: Path):
    claim_lock(tmp_path, "src/main.py", session_id="session-a")

    claim = check_lock(tmp_path, "src/main.py")

    assert claim is not None
    assert claim.session_id == "session-a"
    assert claim.path == "src/main.py"


def test_claim_creates_locks_dir_if_missing(tmp_path: Path):
    locks_dir = tmp_path / "does" / "not" / "exist"
    claim_lock(locks_dir, "a.py", session_id="s1")
    assert locks_dir.exists()


def test_different_paths_get_independent_claims(tmp_path: Path):
    claim_lock(tmp_path, "a.py", session_id="session-a")
    claim_lock(tmp_path, "b.py", session_id="session-b")

    assert check_lock(tmp_path, "a.py").session_id == "session-a"
    assert check_lock(tmp_path, "b.py").session_id == "session-b"


def test_re_claim_by_different_session_overwrites(tmp_path: Path):
    claim_lock(tmp_path, "a.py", session_id="session-a")
    claim_lock(tmp_path, "a.py", session_id="session-b")

    assert check_lock(tmp_path, "a.py").session_id == "session-b"


def test_expired_claim_is_treated_as_absent(tmp_path: Path):
    claim_lock(tmp_path, "a.py", session_id="session-a")
    # Manually age the claim past the TTL rather than sleeping in a test.
    lock_files = list(tmp_path.glob("*.lock"))
    assert len(lock_files) == 1
    record = json.loads(lock_files[0].read_text(encoding="utf-8"))
    record["timestamp"] = time.time() - (LOCK_TTL_SECONDS + 1)
    lock_files[0].write_text(json.dumps(record), encoding="utf-8")

    assert check_lock(tmp_path, "a.py") is None


def test_claim_just_under_ttl_is_still_live(tmp_path: Path):
    claim_lock(tmp_path, "a.py", session_id="session-a")
    lock_files = list(tmp_path.glob("*.lock"))
    record = json.loads(lock_files[0].read_text(encoding="utf-8"))
    record["timestamp"] = time.time() - (LOCK_TTL_SECONDS - 5)
    lock_files[0].write_text(json.dumps(record), encoding="utf-8")

    assert check_lock(tmp_path, "a.py") is not None


def test_custom_ttl_is_respected(tmp_path: Path):
    claim_lock(tmp_path, "a.py", session_id="session-a")
    lock_files = list(tmp_path.glob("*.lock"))
    record = json.loads(lock_files[0].read_text(encoding="utf-8"))
    record["timestamp"] = time.time() - 10
    lock_files[0].write_text(json.dumps(record), encoding="utf-8")

    assert check_lock(tmp_path, "a.py", ttl_seconds=5) is None
    assert check_lock(tmp_path, "a.py", ttl_seconds=60) is not None


def test_corrupt_lock_file_fails_open_not_blocking(tmp_path: Path):
    """A malformed lock file must never permanently wedge a file --
    that would be strictly worse than having no lock mechanism at all."""
    claim_lock(tmp_path, "a.py", session_id="session-a")
    lock_files = list(tmp_path.glob("*.lock"))
    lock_files[0].write_text("{not valid json at all", encoding="utf-8")

    assert check_lock(tmp_path, "a.py") is None


def test_lock_file_missing_expected_keys_fails_open(tmp_path: Path):
    claim_lock(tmp_path, "a.py", session_id="session-a")
    lock_files = list(tmp_path.glob("*.lock"))
    lock_files[0].write_text(json.dumps({"unexpected": "shape"}), encoding="utf-8")

    assert check_lock(tmp_path, "a.py") is None


def test_lock_claim_age_seconds_is_nonnegative_and_recent():
    claim = LockClaim(path="a.py", session_id="s1", timestamp=time.time())
    assert 0 <= claim.age_seconds < 1
