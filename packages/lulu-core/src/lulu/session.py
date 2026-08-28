"""session.py: turn-by-turn persistence and resume.

A session is a JSONL file at .lulu/logs/sessions/<id>.jsonl -- one record
per message, append-only (crash-safe, no schema migration, grep-able).
Resuming a session means replaying that file back into a Message history,
not keeping separate in-memory state that could drift from what's on
disk. Same pattern as RoutingTrace and permissions.jsonl: the log IS the
state, not a side effect of it.
"""

from __future__ import annotations

import dataclasses
import json
import time
import uuid
from pathlib import Path

from lulu.llm.client import Message, ToolCall, ToolResult, Usage
from lulu.loop import TurnResult


class Session:
    def __init__(self, session_id: str, log_dir: Path) -> None:
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
