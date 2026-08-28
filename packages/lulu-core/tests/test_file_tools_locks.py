"""write_file/edit_file refresh their session's lock claim after writing,
when locks_dir + session_id are provided -- and stay exactly as they were
before (no claiming, no behavior change) when they're not, so every
pre-existing file_tools test keeps passing unmodified."""

from __future__ import annotations

from pathlib import Path

import pytest

from lulu.locks import check_lock
from lulu.tools.file_tools import make_edit_file_tool, make_write_file_tool


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    return project


def test_write_file_claims_lock_when_configured(root: Path, tmp_path: Path):
    locks_dir = tmp_path / "locks"
    tool = make_write_file_tool(root, locks_dir=locks_dir, session_id="session-a")

    tool.handler({"path": "a.py", "content": "x = 1"})

    claim = check_lock(locks_dir, "a.py")
    assert claim is not None
    assert claim.session_id == "session-a"


def test_write_file_does_not_claim_without_locks_dir(root: Path):
    tool = make_write_file_tool(root)  # no locks_dir, no session_id
    result = tool.handler({"path": "a.py", "content": "x = 1"})
    assert "wrote" in result  # behaves exactly as before this feature


def test_edit_file_claims_lock_when_configured(root: Path, tmp_path: Path):
    (root / "a.py").write_bytes(b"x = 1\n")
    locks_dir = tmp_path / "locks"
    tool = make_edit_file_tool(root, locks_dir=locks_dir, session_id="session-a")

    tool.handler({"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"})

    claim = check_lock(locks_dir, "a.py")
    assert claim is not None
    assert claim.session_id == "session-a"


def test_edit_file_does_not_claim_without_session_id(root: Path, tmp_path: Path):
    (root / "a.py").write_bytes(b"x = 1\n")
    locks_dir = tmp_path / "locks"
    tool = make_edit_file_tool(root, locks_dir=locks_dir, session_id=None)

    tool.handler({"path": "a.py", "old_string": "x = 1", "new_string": "x = 2"})

    assert not locks_dir.exists() or list(locks_dir.glob("*.lock")) == []
