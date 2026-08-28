"""Adversarial coverage for the one function every file tool's safety
depends on: resolve_within_root. Each test here is "how would I try to
read/write outside the project root if I were a malicious or just
confused tool call."
"""

from __future__ import annotations

from pathlib import Path

import pytest

from lulu.tools.sandbox import PathEscapesRootError, is_within_root, resolve_within_root


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    (project / "src" / "main.py").write_text("print('hi')")
    return project


def test_relative_path_within_root_resolves(root: Path):
    resolved = resolve_within_root(root, "src/main.py")
    assert resolved == (root / "src" / "main.py").resolve()


def test_root_itself_is_allowed(root: Path):
    resolved = resolve_within_root(root, ".")
    assert resolved == root.resolve()


def test_new_file_that_does_not_exist_yet_still_resolves(root: Path):
    resolved = resolve_within_root(root, "src/new_file.py")
    assert resolved == (root / "src" / "new_file.py").resolve()


def test_simple_dotdot_traversal_is_blocked(root: Path):
    with pytest.raises(PathEscapesRootError):
        resolve_within_root(root, "../secret.txt")


def test_nested_dotdot_traversal_is_blocked(root: Path):
    with pytest.raises(PathEscapesRootError):
        resolve_within_root(root, "src/../../secret.txt")


def test_deep_dotdot_traversal_is_blocked(root: Path):
    with pytest.raises(PathEscapesRootError):
        resolve_within_root(root, "../../../../../../etc/passwd")


def test_dotdot_that_returns_back_inside_root_is_allowed(root: Path):
    """Goes up and back down further than root, but the *final* resolved
    location is still inside root -- this must be allowed, not just any
    path containing '..' rejected, since that would make legitimate
    relative paths from deep subdirectories unusable."""
    (root / "a" / "b").mkdir(parents=True)
    resolved = resolve_within_root(root, "a/b/../../src/main.py")
    assert resolved == (root / "src" / "main.py").resolve()


def test_absolute_path_outside_root_is_blocked(root: Path, tmp_path: Path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret")
    with pytest.raises(PathEscapesRootError):
        resolve_within_root(root, str(outside))


def test_absolute_path_to_a_sibling_directory_sharing_a_prefix_is_blocked(root: Path):
    """root is .../project -- a sibling literally named .../project-evil
    must not pass a naive string-prefix check ('startswith root')."""
    sibling = root.parent / "project-evil"
    sibling.mkdir()
    evil_file = sibling / "x.txt"
    with pytest.raises(PathEscapesRootError):
        resolve_within_root(root, str(evil_file))


def test_windows_style_backslash_traversal_is_blocked(root: Path):
    with pytest.raises(PathEscapesRootError):
        resolve_within_root(root, "..\\..\\secret.txt")


def test_mixed_separator_traversal_is_blocked(root: Path):
    with pytest.raises(PathEscapesRootError):
        resolve_within_root(root, "src/..\\../secret.txt")


def test_absolute_windows_system_path_is_blocked(root: Path):
    with pytest.raises(PathEscapesRootError):
        resolve_within_root(root, "C:\\Windows\\System32\\config\\SAM")


def test_trailing_slash_within_root_is_allowed(root: Path):
    resolved = resolve_within_root(root, "src/")
    assert resolved == (root / "src").resolve()


def test_current_dir_reference_within_root_is_allowed(root: Path):
    resolved = resolve_within_root(root, "./src/./main.py")
    assert resolved == (root / "src" / "main.py").resolve()


def test_is_within_root_true_for_nested_path(root: Path):
    assert is_within_root(root, root / "src" / "main.py") is True


def test_is_within_root_true_for_root_itself(root: Path):
    assert is_within_root(root, root) is True


def test_is_within_root_false_for_glob_style_traversal(root: Path, tmp_path: Path):
    """Path.glob() will itself return matches outside root when the
    pattern contains '..' (empirically confirmed -- root.glob("../x")
    resolves and matches outside root without error). is_within_root is
    the filter the glob/grep tools apply to every match for exactly this
    reason."""
    outside = tmp_path / "secret.txt"
    outside.write_text("secret")
    leaked_match = root / ".." / "secret.txt"
    assert is_within_root(root, leaked_match) is False
