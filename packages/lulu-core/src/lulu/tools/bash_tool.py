"""Bash tool: runs a shell command with the project root as cwd.

No sandboxing of the command string itself -- see sandbox.py's module
docstring for why that's a deliberate non-goal (string-inspecting shell
commands for safety is a well-known losing game). The real boundary is
permissions.py: bash calls default to "ask", and blast-radius detection
(rm -rf, curl/wget, sudo, redirection outside root) forces "ask"
regardless of AttentionMode, no matter what this tool does internally.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from lulu.tools.base import Tool

DEFAULT_TIMEOUT_S = 120
MAX_TIMEOUT_S = 600
MAX_OUTPUT_CHARS = 30_000


def make_bash_tool(root: Path) -> Tool:
    def handler(args: dict) -> str:
        command = args["command"]
        timeout_s = min(float(args.get("timeout_s", DEFAULT_TIMEOUT_S)), MAX_TIMEOUT_S)

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"command timed out after {timeout_s}s") from exc

        output = (proc.stdout or "") + (proc.stderr or "")
        if len(output) > MAX_OUTPUT_CHARS:
            omitted = len(output) - MAX_OUTPUT_CHARS
            output = output[:MAX_OUTPUT_CHARS] + f"\n...[truncated, {omitted} more chars]"

        return f"exit code: {proc.returncode}\n" + (output or "(no output)")

    return Tool(
        name="bash",
        description=(
            "Run a shell command in the project root. Not sandboxed at the "
            "command level -- the permission system decides what's allowed to run."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "timeout_s": {
                    "type": "number",
                    "description": f"defaults to {DEFAULT_TIMEOUT_S}s, capped at {MAX_TIMEOUT_S}s",
                },
            },
            "required": ["command"],
        },
        handler=handler,
    )
