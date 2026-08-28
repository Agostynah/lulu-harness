"""Bash tool: exit code surfacing, cwd scoping, output truncation, and
timeout handling. Uses `sys.executable -c "..."` for test commands so
these pass identically whether the underlying shell is cmd.exe (Windows,
what shell=True actually invokes here) or /bin/sh (POSIX) -- no shell-
specific syntax, just program + args.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from lulu.tools.bash_tool import make_bash_tool

# sys.executable can contain spaces (this repo lives under "Lulu Harness"),
# which breaks unquoted under cmd.exe (shell=True's Windows shell) --
# quoting is what test_captures_stdout_and_exit_code etc. actually caught.
PY = f'"{sys.executable}"'


@pytest.fixture
def root(tmp_path: Path) -> Path:
    project = tmp_path / "project"
    project.mkdir()
    return project


def test_captures_stdout_and_exit_code(root: Path):
    tool = make_bash_tool(root)
    result = tool.handler({"command": f'{PY} -c "print(42)"'})
    assert "exit code: 0" in result
    assert "42" in result


def test_nonzero_exit_code_is_surfaced_not_raised(root: Path):
    tool = make_bash_tool(root)
    result = tool.handler({"command": f'{PY} -c "import sys; sys.exit(3)"'})
    assert "exit code: 3" in result


def test_runs_with_project_root_as_cwd(root: Path):
    tool = make_bash_tool(root)
    result = tool.handler({"command": f'{PY} -c "import os; print(os.getcwd())"'})
    assert str(root.resolve()) in result


def test_captures_stderr_too(root: Path):
    tool = make_bash_tool(root)
    result = tool.handler(
        {"command": f'{PY} -c "import sys; sys.stderr.write(chr(69)+chr(82)+chr(82))"'}
    )
    assert "ERR" in result


def test_output_over_limit_is_truncated(root: Path, monkeypatch: pytest.MonkeyPatch):
    import lulu.tools.bash_tool as bt

    monkeypatch.setattr(bt, "MAX_OUTPUT_CHARS", 20)
    tool = make_bash_tool(root)
    result = tool.handler({"command": f'{PY} -c "print(\'x\' * 500)"'})
    assert "truncated" in result


def test_timeout_raises_timeout_error(root: Path):
    tool = make_bash_tool(root)
    with pytest.raises(TimeoutError):
        tool.handler(
            {"command": f'{PY} -c "import time; time.sleep(5)"', "timeout_s": 0.2}
        )


def test_requested_timeout_is_capped(root: Path, monkeypatch: pytest.MonkeyPatch):
    """An agent (or a prompt-injected instruction) asking for an absurdly
    long timeout shouldn't be able to hang the harness indefinitely."""
    import lulu.tools.bash_tool as bt

    monkeypatch.setattr(bt, "MAX_TIMEOUT_S", 0.2)
    tool = make_bash_tool(root)
    with pytest.raises(TimeoutError):
        tool.handler(
            {"command": f'{PY} -c "import time; time.sleep(5)"', "timeout_s": 999999}
        )
