"""session.py: turn-by-turn persistence and resume.

A session is a JSONL file at .lulu/logs/sessions/<id>.jsonl -- one record
per message, append-only (crash-safe, no schema migration, grep-able).
Resuming a session means replaying that file back into a Message history,
not keeping separate in-memory state that could drift from what's on
disk. Same pattern as RoutingTrace and permissions.jsonl: the log IS the
state, not a side effect of it.

session_id is validated in __init__, not just at server.py's HTTP
boundary. Caught by adversarial review, not assumed safe: server.py
accepts session_id as a caller-supplied, unauthenticated query parameter
and this class used to interpolate it directly into a filesystem path
with no traversal or absolute-path check -- `?session_id=../../../etc/
cron.d/evil` would happily make `self.path` point anywhere on disk, and
the next write would create/overwrite a file there. Validating in
Session.__init__ (the one place every caller, present and future, HTTP
or otherwise, necessarily goes through) is the sound fix, the same
"validate at the lowest common point" principle sandbox.py already uses
for file paths -- not a check bolted onto server.py alone, which would
protect this one caller and nothing else that constructs a Session
directly.
"""

from __future__ import annotations

import dataclasses
import json
import re
import time
import uuid
from pathlib import Path

from lulu.llm.client import Message, ToolCall, ToolResult, Usage
from lulu.loop import TurnResult

# Exactly the alphabet Session.new() itself produces (uuid4 hex), with
# some room for length -- hex characters can never form a path-traversal
# sequence ("..", "/", "\", a drive letter, a null byte), so this is a
# sound allowlist, not a blocklist trying to catch every bad pattern.
_VALID_SESSION_ID = re.compile(r"^[0-9a-f]{1,64}$")


class InvalidSessionIdError(ValueError):
    pass


def _validate_session_id(session_id: str) -> None:
    if not _VALID_SESSION_ID.match(session_id):
        raise InvalidSessionIdError(
            f"invalid session_id {session_id!r}: must match {_VALID_SESSION_ID.pattern}"
        )


class Session:
    def __init__(self, session_id: str, log_dir: Path) -> None:
        _validate_session_id(session_id)
        self.session_id = session_id
        self.log_dir = log_dir
        self.path = log_dir / f"{session_id}.jsonl"

    @classmethod
    def new(cls, log_dir: Path) -> "Session":
        return cls(session_id=uuid.uuid4().hex[:12], log_dir=log_dir)

    @classmethod
    def resume(cls, session_id: str, log_dir: Path) -> "Session":
        return cls(session_id=session_id, log_dir=log_dir)

    @classmethod
    def list_sessions(cls, log_dir: Path) -> list[str]:
        if not log_dir.exists():
            return []
        return sorted(p.stem for p in log_dir.glob("*.jsonl"))

    def append_message(self, message: Message, usage: Usage | None = None) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "role": message.role,
            "content": message.content,
            "tool_calls": [dataclasses.asdict(tc) for tc in message.tool_calls],
            "tool_results": [dataclasses.asdict(tr) for tr in message.tool_results],
            "usage": dataclasses.asdict(usage) if usage is not None else None,
            "timestamp": time.time(),
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def append_turn_result(self, result: TurnResult, messages_before: int) -> None:
        """Appends only the messages this turn actually added (everything
        past `messages_before`, the length of the history passed into
        run_turn), attributing each queued Usage to the assistant message
        that produced it -- they line up 1:1 in call order."""
        usage_iter = iter(result.usages)
        for message in result.messages[messages_before:]:
            usage = next(usage_iter, None) if message.role == "assistant" else None
            self.append_message(message, usage=usage)

    def load_history(self) -> list[Message]:
        # O(n) in the session's total message count on every call, not
        # incremental -- re-reads and re-parses the whole JSONL file from
        # byte 0 each time (same for total_usage() below). Fine at
        # harness scale: a session is one human's conversation, not a
        # shared log, so "total messages" stays small and this runs once
        # per turn, not in a hot loop. The honest fix if that stops being
        # true is tailing from a saved byte offset instead of reparsing
        # everything; not worth the complexity at today's scale.
        if not self.path.exists():
            return []
        messages: list[Message] = []
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                messages.append(
                    Message(
                        role=record["role"],
                        content=record.get("content", ""),
                        tool_calls=[ToolCall(**tc) for tc in record.get("tool_calls", [])],
                        tool_results=[ToolResult(**tr) for tr in record.get("tool_results", [])],
                    )
                )
        return messages

    def total_usage(self) -> Usage:
        total = Usage()
        if not self.path.exists():
            return total
        with self.path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                usage = json.loads(line).get("usage")
                if usage:
                    total.input_tokens += usage.get("input_tokens", 0)
                    total.output_tokens += usage.get("output_tokens", 0)
        return total
