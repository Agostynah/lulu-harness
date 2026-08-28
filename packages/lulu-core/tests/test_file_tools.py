"""File tools: happy paths plus adversarial path-traversal attempts
against every tool that touches disk. read/write/edit are expected to
raise PathEscapesRootError; glob/grep are expected to silently exclude
anything outside root rather than error (a glob pattern isn't inherently
malicious the way an explicit path is -- it's just filtered)."""

from __future__ import annotations

from pathlib import Path

import pytest

from lulu.tools.file_tools import (
    make_edit_file_tool,
    make_glob_tool,
    make_grep_tool,
    make_read_file_tool,
    make_write_file_tool,
)
from lulu.tools.sandbox import PathEscapesRootError


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    # write_bytes, not write_text: Path.write_text on Windows translates
    # \n -> \r\n, which would make the exact-content assertions below
    # (and read_file, which reads raw bytes with no newline normalization
    # of its own -- deliberately, to not silently rewrite a file's line
    # endings) disagree with what was actually written.
    (project / "src" / "main.py").write_bytes(b"print('hello')\nprint('hello')\n")
    return project


# --- read_file ---


def test_read_file_returns_contents(root: Path):
    tool = make_read_file_tool(root)
    assert tool.handler({"path": "src/main.py"}) == "print('hello')\nprint('hello')\n"


def test_read_file_missing_raises(root: Path):
    tool = make_read_file_tool(root)
    with pytest.raises(FileNotFoundError):
        tool.handler({"path": "src/nope.py"})


def test_read_file_directory_raises(root: Path):
    tool = make_read_file_tool(root)
    with pytest.raises(IsADirectoryError):
        tool.handler({"path": "src"})


def test_read_file_blocks_traversal(root: Path):
    tool = make_read_file_tool(root)
    with pytest.raises(PathEscapesRootError):
        tool.handler({"path": "../../etc/passwd"})


def test_read_file_respects_size_limit(root: Path, monkeypatch: pytest.MonkeyPatch):
    import lulu.tools.file_tools as ft

    monkeypatch.setattr(ft, "MAX_READ_BYTES", 5)
    tool = make_read_file_tool(root)
    with pytest.raises(ValueError, match="byte read limit"):
        tool.handler({"path": "src/main.py"})


# --- write_file ---


def test_write_file_creates_new_file(root: Path):
    tool = make_write_file_tool(root)
    tool.handler({"path": "src/new.py", "content": "x = 1"})
    assert (root / "src" / "new.py").read_text() == "x = 1"


def test_write_file_creates_parent_dirs(root: Path):
    tool = make_write_file_tool(root)
    tool.handler({"path": "a/b/c/deep.py", "content": "ok"})
    assert (root / "a" / "b" / "c" / "deep.py").read_text() == "ok"


def test_write_file_blocks_traversal(root: Path):
    tool = make_write_file_tool(root)
    with pytest.raises(PathEscapesRootError):
        tool.handler({"path": "../outside.py", "content": "malicious"})


def test_write_file_blocks_absolute_path_outside_root(root: Path, tmp_path: Path):
    tool = make_write_file_tool(root)
    outside = tmp_path / "outside.py"
    with pytest.raises(PathEscapesRootError):
        tool.handler({"path": str(outside), "content": "malicious"})
    assert not outside.exists()


# --- edit_file ---


def test_edit_file_replaces_unique_match(root: Path):
    (root / "unique.py").write_bytes(b"x = 1\ny = 2\n")
    tool = make_edit_file_tool(root)
    tool.handler({"path": "unique.py", "old_string": "x = 1", "new_string": "x = 99"})
    assert (root / "unique.py").read_bytes() == b"x = 99\ny = 2\n"


def test_edit_file_ambiguous_match_without_replace_all_raises(root: Path):
    tool = make_edit_file_tool(root)
    with pytest.raises(ValueError, match="not unique"):
        tool.handler({"path": "src/main.py", "old_string": "print('hello')", "new_string": "print('bye')"})


def test_edit_file_replace_all_replaces_every_occurrence(root: Path):
    tool = make_edit_file_tool(root)
    tool.handler(
        {"path": "src/main.py", "old_string": "print('hello')", "new_string": "print('bye')", "replace_all": True}
    )
    assert (root / "src" / "main.py").read_text() == "print('bye')\nprint('bye')\n"


def test_edit_file_missing_old_string_raises(root: Path):
    tool = make_edit_file_tool(root)
    with pytest.raises(ValueError, match="not found"):
        tool.handler({"path": "src/main.py", "old_string": "does not exist", "new_string": "x"})


def test_edit_file_blocks_traversal(root: Path):
    tool = make_edit_file_tool(root)
    with pytest.raises(PathEscapesRootError):
        tool.handler({"path": "../../outside.py", "old_string": "a", "new_string": "b"})


def test_edit_file_does_not_touch_untouched_lines_endings(root: Path):
    """Regression: editing one line must not rewrite every other line's
    ending in the file. On Windows, Path.write_text without newline=""
    silently converts every \\n to \\r\\n on write, which would turn a
    one-line edit into a diff touching the entire file."""
    (root / "multi.py").write_bytes(b"a = 1\nb = 2\nc = 3\n")
    tool = make_edit_file_tool(root)
    tool.handler({"path": "multi.py", "old_string": "b = 2", "new_string": "b = 200"})
    assert (root / "multi.py").read_bytes() == b"a = 1\nb = 200\nc = 3\n"


def test_write_file_preserves_exact_line_endings(root: Path):
    tool = make_write_file_tool(root)
    tool.handler({"path": "exact.py", "content": "line1\nline2\n"})
    assert (root / "exact.py").read_bytes() == b"line1\nline2\n"


# --- glob ---


def test_glob_finds_matching_files(root: Path):
    tool = make_glob_tool(root)
    result = tool.handler({"pattern": "**/*.py"})
    assert "src/main.py" in result.replace("\\", "/")


def test_glob_no_matches_returns_placeholder(root: Path):
    tool = make_glob_tool(root)
    assert tool.handler({"pattern": "**/*.rs"}) == "(no matches)"


def test_glob_traversal_pattern_excludes_outside_matches(root: Path, tmp_path: Path):
    """root.glob('../secret.txt') DOES resolve outside root (confirmed in
    test_sandbox.py) -- the tool must filter that out, not return it."""
    outside = tmp_path / "secret.txt"
    outside.write_text("leak me")
    tool = make_glob_tool(root)
    result = tool.handler({"pattern": "../*.txt"})
    assert "secret.txt" not in result


# --- grep ---


def test_grep_finds_matching_lines(root: Path):
    tool = make_grep_tool(root)
    result = tool.handler({"pattern": "hello"})
    assert "src/main.py" in result.replace("\\", "/")
    assert result.count("hello") == 2


def test_grep_scoped_by_glob(root: Path):
    (root / "notes.txt").write_text("hello from notes\n")
    tool = make_grep_tool(root)
    result = tool.handler({"pattern": "hello", "glob": "**/*.py"})
    assert "notes.txt" not in result


def test_grep_no_matches_returns_placeholder(root: Path):
    tool = make_grep_tool(root)
    assert tool.handler({"pattern": "nonexistent_string_xyz"}) == "(no matches)"


def test_grep_invalid_regex_raises(root: Path):
    tool = make_grep_tool(root)
    with pytest.raises(ValueError, match="invalid regex"):
        tool.handler({"pattern": "["})


def test_grep_traversal_pattern_excludes_outside_matches(root: Path, tmp_path: Path):
    outside = tmp_path / "secret.txt"
    outside.write_text("hello secret\n")
    tool = make_grep_tool(root)
    result = tool.handler({"pattern": "hello", "glob": "../*.txt"})
    assert "secret" not in result
