"""File tools: read/write/edit/glob/grep, all sandboxed to the project
root. read/write/edit go through resolve_within_root (see sandbox.py).
glob/grep additionally filter every result through is_within_root, because
Path.glob() itself will happily return matches outside root when the glob
*pattern* contains `..` -- confirmed empirically, not a hypothetical
(root.glob("../secret.txt") resolves and matches outside root with no
error). Sandboxing the pattern isn't enough; the results have to be
checked too.
"""

from __future__ import annotations

import re
from pathlib import Path

from lulu.locks import claim_lock
from lulu.tools.base import Tool
from lulu.tools.sandbox import is_within_root, resolve_within_root

MAX_READ_BYTES = 500_000
MAX_MATCHES = 200


def make_read_file_tool(root: Path) -> Tool:
    def handler(args: dict) -> str:
        path = resolve_within_root(root, args["path"])
        if not path.exists():
            raise FileNotFoundError(f"{args['path']} does not exist")
        if path.is_dir():
            raise IsADirectoryError(f"{args['path']} is a directory, not a file")
        data = path.read_bytes()
        if len(data) > MAX_READ_BYTES:
            raise ValueError(
                f"{args['path']} is {len(data):,} bytes, over the {MAX_READ_BYTES:,}-byte read limit"
            )
        return data.decode("utf-8", errors="replace")

    return Tool(
        name="read_file",
        description="Read a text file within the project.",
        input_schema={
            "type": "object",
            "properties": {"path": {"type": "string", "description": "Path relative to the project root"}},
            "required": ["path"],
        },
        handler=handler,
    )


def make_write_file_tool(
    root: Path, locks_dir: Path | None = None, session_id: str | None = None
) -> Tool:
    def handler(args: dict) -> str:
        path = resolve_within_root(root, args["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if locks_dir is not None and session_id is not None:
            # Claim BEFORE writing, not after: see locks.py and
            # permissions.py's LOCKABLE_TOOLS check, which reads this claim
            # before a competing session's write. Claiming after the write
            # left a window where the file was already mutated but no
            # claim existed yet for a concurrent session's check_lock() to
            # see -- narrower is better even though this is advisory, not
            # a hard mutex (see locks.py's module docstring).
            claim_lock(locks_dir, args["path"], session_id)
        # newline="": write exactly the bytes implied by `content`, no
        # platform-specific \n -> \r\n translation. Same reasoning as
        # edit_file -- a file tool shouldn't silently transform content
        # the caller gave it verbatim.
        path.write_text(args["content"], encoding="utf-8", newline="")
        return f"wrote {len(args['content'])} chars to {args['path']}"

    return Tool(
        name="write_file",
        description="Write (overwrite) a text file within the project, creating parent directories as needed.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
        handler=handler,
    )


def make_edit_file_tool(
    root: Path, locks_dir: Path | None = None, session_id: str | None = None
) -> Tool:
    def handler(args: dict) -> str:
        path = resolve_within_root(root, args["path"])
        if not path.exists():
            raise FileNotFoundError(f"{args['path']} does not exist")
        # newline="" disables Python's universal-newline translation on
        # both ends. Without it, read_text/write_text silently normalize
        # every line ending in the file to the platform default (CRLF on
        # Windows) on every edit, even ones nowhere near the touched
        # line -- which shows up as a massive unwanted diff in git for a
        # one-line change. A code-editing tool must never do that.
        # (Path.read_text() only gained a `newline` param in 3.13; this
        # project targets 3.11+, so open() directly for the read side.)
        with path.open(encoding="utf-8", newline="") as f:
            text = f.read()
        old, new = args["old_string"], args["new_string"]
        replace_all = bool(args.get("replace_all", False))
        count = text.count(old)
        if count == 0:
            raise ValueError(f"old_string not found in {args['path']}")
        if count > 1 and not replace_all:
            raise ValueError(
                f"old_string is not unique in {args['path']} ({count} occurrences); "
                "pass replace_all=true or a more specific old_string"
            )
        replaced = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        if locks_dir is not None and session_id is not None:
            # Claim BEFORE writing -- see make_write_file_tool's handler
            # above for why.
            claim_lock(locks_dir, args["path"], session_id)
        path.write_text(replaced, encoding="utf-8", newline="")
        n = count if replace_all else 1
        return f"replaced {n} occurrence(s) in {args['path']}"

    return Tool(
        name="edit_file",
        description="Replace an exact string match in a file. Fails if old_string is not unique unless replace_all is set.",
        input_schema={
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_string": {"type": "string"},
                "new_string": {"type": "string"},
                "replace_all": {"type": "boolean"},
            },
            "required": ["path", "old_string", "new_string"],
        },
        handler=handler,
    )


def make_glob_tool(root: Path) -> Tool:
    def handler(args: dict) -> str:
        pattern = args["pattern"]
        root_resolved = root.resolve()
        matches = sorted(
            str(p.relative_to(root_resolved))
            for p in root.glob(pattern)
            if p.is_file() and is_within_root(root, p)
        )
        return "\n".join(matches) if matches else "(no matches)"

    return Tool(
        name="glob",
        description="Find files under the project root matching a glob pattern (e.g. '**/*.py').",
        input_schema={
            "type": "object",
            "properties": {"pattern": {"type": "string"}},
            "required": ["pattern"],
        },
        handler=handler,
    )


def make_grep_tool(root: Path) -> Tool:
    def handler(args: dict) -> str:
        try:
            pattern = re.compile(args["pattern"])
        except re.error as exc:
            raise ValueError(f"invalid regex {args['pattern']!r}: {exc}") from exc
        glob_filter = args.get("glob", "**/*")

        matches: list[str] = []
        for path in root.glob(glob_filter):
            if not path.is_file() or not is_within_root(root, path):
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue
            for i, line in enumerate(text.splitlines(), start=1):
                if pattern.search(line):
                    matches.append(f"{path.relative_to(root.resolve())}:{i}:{line}")
                    if len(matches) >= MAX_MATCHES:
                        break
            if len(matches) >= MAX_MATCHES:
                break

        return "\n".join(matches) if matches else "(no matches)"

    return Tool(
        name="grep",
        description="Search file contents under the project root with a regex pattern, optionally scoped by a glob.",
        input_schema={
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "glob": {"type": "string", "description": "defaults to '**/*'"},
            },
            "required": ["pattern"],
        },
        handler=handler,
    )
