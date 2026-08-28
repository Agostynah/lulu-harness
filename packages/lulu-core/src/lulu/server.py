"""server.py: FastAPI + SSE surface over the same AgentLoop/Session/
MemoryStore wiring cli.py uses. `lulu-server` is a sibling entrypoint to
the `lulu` CLI, not a separate product -- both are thin wrappers over the
same harness.

State model: one shared MemoryStore per server process, deliberately.
MemoryStore is in-process-only (memory.py), and sharing it across
sessions is what lets one session's writes actually inform another
session's retrieval -- a taste of "memory as the channel between
concurrent sessions" (ROADMAP.md) without building the dedicated
mechanism described there. One AgentLoop per session, created lazily and
cached in-process; Session itself is file-backed (session.py), so
conversation history survives a server restart even though memory
doesn't.

Known limitation, stated rather than hidden: the server cannot pause
synchronously to ask a human a permission question over a stateless HTTP
request the way the CLI's `input()` does. `server_ask_human` always
denies -- fail CLOSED, not open. A risky action attempted through the web
UI gets refused, not silently executed and not silently hung waiting for
an answer that can never arrive over this transport. A real two-phase
(or websocket-based) approval flow is ROADMAP.md's job, not this file's.
Point the server at a permissive mode (auto_edits/auto) if you want it to
actually get things done -- blast-radius and lock-conflict detection
still force (and then auto-deny) ASK regardless of mode, same as always.

The /stream endpoint is SSE transport, not true token-level streaming: it
runs a turn to completion synchronously, then emits the result as a
burst of events. Real incremental streaming would need AgentLoop's tool
loop to interleave with ModelClient.stream(), which only AnthropicClient
implements for real today (see llm/openai_compatible.py's docstring for
the same honestly-scoped-down tradeoff made there).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import run_in_threadpool

from lulu.attention import AttentionMode
from lulu.cli import build_model_client, build_tool_registry
from lulu.config import LuluConfig, load_config
from lulu.counterfactual import compute_counterfactuals, savings_pct
from lulu.loop import AgentLoop
from lulu.memory import MemoryStore
from lulu.permissions import PermissionChecker
from lulu.session import InvalidSessionIdError, Session

DEFAULT_SERVER_MODE = AttentionMode.AUTO_EDITS


def server_ask_human(tool_name: str, arguments: dict, reason: str) -> bool:
    """Always denies. See module docstring: there is no synchronous human
    to ask over a stateless HTTP request, and failing open would mean
    silently executing something that was specifically flagged as needing
    approval. The denial reason is visible to the model (and from there,
    the user) via the normal tool_result error path -- it isn't silent,
    it's just not interactive yet."""
    return False


@dataclasses.dataclass
class SessionEntry:
    loop: AgentLoop
    session: Session
    last_trace: Any = None  # RoutingTrace | None -- what /api/sessions/{id}/cost compares against
    last_scope: str | None = None  # the scope that trace was computed under -- /cost must use
    # the SAME scope, not every shard that happens to exist, or it leaks
    # "how much data other scopes have" the same way memory.py's own bug did


class TurnRequest(BaseModel):
    prompt: str
    scope: str | None = None


class LuluServer:
    """Holds the process-wide state (shared memory, per-session loops) so
    it can be constructed fresh per test instead of relying on module-
    level globals -- see tests/test_server.py."""

    def __init__(
        self,
        root: Path,
        config: LuluConfig | None = None,
        model_override: Any = None,
        memory_override: MemoryStore | None = None,
        mode: AttentionMode | None = None,
    ) -> None:
        self.root = root
        self.config = config or load_config(root / "lulu.toml")
        # Deliberately NOT self.config.attention_mode: that field defaults
        # to MANUAL (the right default for the CLI, where ASK works via a
        # real TTY), which combined with server_ask_human's hard-coded
        # denial would make the server silently read-only by default.
        # The server needs its own, separately-defaulted mode.
        self.mode = mode or DEFAULT_SERVER_MODE
        self.log_dir = root / ".lulu" / "logs"
        self.locks_dir = root / ".lulu" / "locks"
        self.memory = memory_override or MemoryStore()
        self._model_override = model_override
        self._sessions: dict[str, SessionEntry] = {}

    def _build_loop(self, session_id: str) -> AgentLoop:
        model = self._model_override or build_model_client(self.config)
        tools = build_tool_registry(self.root, locks_dir=self.locks_dir, session_id=session_id)
        permissions = PermissionChecker(
            mode=self.mode,
            log_path=self.log_dir / "permissions.jsonl",
            locks_dir=self.locks_dir,
            session_id=session_id,
        )
        return AgentLoop(
            model=model,
            tools=tools,
            permissions=permissions,
            ask_human=server_ask_human,
            max_iterations=self.config.max_iterations,
            memory=self.memory,
        )

    def get_or_create_session(self, session_id: str | None) -> SessionEntry:
        if session_id and session_id in self._sessions:
            return self._sessions[session_id]

        session = (
            Session.resume(session_id, self.log_dir / "sessions")
            if session_id
            else Session.new(self.log_dir / "sessions")
        )
        entry = SessionEntry(loop=self._build_loop(session.session_id), session=session)
        self._sessions[session.session_id] = entry
        return entry


def create_app(
    root: Path,
    config: LuluConfig | None = None,
    model_override: Any = None,
    memory_override: MemoryStore | None = None,
    mode: AttentionMode | None = None,
) -> FastAPI:
    state = LuluServer(
        root=root, config=config, model_override=model_override, memory_override=memory_override, mode=mode
    )
    app = FastAPI(title="Lulu")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(InvalidSessionIdError)
    async def _invalid_session_id(_request, exc: InvalidSessionIdError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    app.state.lulu = state

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/config")
    def get_config() -> dict[str, Any]:
        return {
            "provider": state.config.provider,
            "model": state.config.model,
            "attention_mode": state.mode.value,
            "root": str(state.root),
        }

    @app.post("/api/sessions")
    def create_session(session_id: str | None = None) -> dict[str, Any]:
        entry = state.get_or_create_session(session_id)
        return {"session_id": entry.session.session_id, "history": _serialize_history(entry.session)}

    @app.get("/api/sessions/{session_id}/history")
    def get_history(session_id: str) -> dict[str, Any]:
        entry = state._sessions.get(session_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
        return {"session_id": session_id, "history": _serialize_history(entry.session)}

    @app.post("/api/sessions/{session_id}/turn")
    def run_turn(session_id: str, body: TurnRequest) -> dict[str, Any]:
        entry = state.get_or_create_session(session_id)
        history = entry.session.load_history()
        messages_before = len(history)
        loop = entry.loop
        loop.memory_scope = body.scope
        result = loop.run_turn(history, body.prompt)
        entry.session.append_turn_result(result, messages_before)
        entry.last_trace = result.trace
        entry.last_scope = body.scope
        return _serialize_turn_result(result)

    @app.get("/api/sessions/{session_id}/cost")
    def get_cost(session_id: str) -> dict[str, Any]:
        """Counterfactual: what query_all/flat_topk WOULD have cost for
        the *same shards*, computed without re-running retrieval -- see
        counterfactual.py. This is what makes the /cost panel's headline
        number possible on every turn, not just when an eval sweep runs."""
        entry = state._sessions.get(session_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown session {session_id!r}")
        if entry.last_trace is None:
            raise HTTPException(status_code=404, detail="no turn has run yet in this session")

        shards = state.memory.shards_for_scope(entry.last_scope)
        counterfactuals = compute_counterfactuals(shards, state.memory.k)
        spent = entry.last_trace.spent

        return {
            "spent": dataclasses.asdict(spent),
            "counterfactuals": [
                {
                    "label": cf.label,
                    "cost": dataclasses.asdict(cf.cost),
                    "tokens_saved_pct": savings_pct(spent, cf.cost, attr="tokens"),
                }
                for cf in counterfactuals
            ],
        }

    @app.get("/api/sessions/{session_id}/turn/stream")
    async def stream_turn(session_id: str, prompt: str, scope: str | None = None):
        entry = state.get_or_create_session(session_id)
        history = entry.session.load_history()
        messages_before = len(history)
        loop = entry.loop
        loop.memory_scope = scope
        # run_in_threadpool, not a direct call: this endpoint is `async
        # def` (required for EventSourceResponse), and loop.run_turn() is
        # synchronous and can block for the model call's full duration --
        # called directly, it would freeze uvicorn's single event loop for
        # every OTHER concurrent session too. Also a real latent crash,
        # not just a performance concern: once a shard ever wraps an MCP
        # connector (connectors/mcp.py), that store's search() calls
        # asyncio.run() internally, which raises immediately if it's
        # already executing inside a running event loop -- exactly the
        # situation a direct call from here would create.
        result = await run_in_threadpool(loop.run_turn, history, prompt)
        entry.session.append_turn_result(result, messages_before)
        entry.last_trace = result.trace
        entry.last_scope = scope
        payload = _serialize_turn_result(result)

        async def event_generator():
            if payload["trace"] is not None:
                yield {"event": "trace", "data": _json(payload["trace"])}
            yield {"event": "message", "data": _json({"content": payload["final_text"]})}
            yield {"event": "usage", "data": _json(payload["usage_totals"])}
            yield {"event": "done", "data": "{}"}

        return EventSourceResponse(event_generator())

    return app


def _serialize_history(session: Session) -> list[dict[str, Any]]:
    return [
        {
            "role": m.role,
            "content": m.content,
            "tool_calls": [dataclasses.asdict(tc) for tc in m.tool_calls],
            "tool_results": [dataclasses.asdict(tr) for tr in m.tool_results],
        }
        for m in session.load_history()
    ]


def _serialize_turn_result(result: Any) -> dict[str, Any]:
    final_text = ""
    for message in reversed(result.messages):
        if message.role == "assistant" and message.content:
            final_text = message.content
            break

    return {
        "final_text": final_text,
        "stopped_reason": result.stopped_reason,
        "iterations": result.iterations,
        "trace": dataclasses.asdict(result.trace) if result.trace is not None else None,
        "usage_totals": {
            "input_tokens": sum(u.input_tokens for u in result.usages),
            "output_tokens": sum(u.output_tokens for u in result.usages),
        },
    }


def _json(data: Any) -> str:
    import json

    return json.dumps(data)


def run() -> None:
    """Entrypoint for `lulu-server` (see pyproject.toml's [project.scripts])."""
    import argparse
    import sys

    import uvicorn

    parser = argparse.ArgumentParser(prog="lulu-server")
    parser.add_argument("--root", default=".", help="Project root (default: cwd)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument(
        "--mode",
        choices=[m.value for m in AttentionMode],
        default=None,
        help=f"Defaults to {DEFAULT_SERVER_MODE.value!r} -- server_ask_human always denies ASK, "
        "so MANUAL would make the server silently read-only.",
    )
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        # No auth layer exists on this server at all (see module
        # docstring's fail-closed-ASK design, which is about permission
        # tiers, not identity). 127.0.0.1 makes that a non-issue -- only a
        # process on this machine can reach it. Anything else, including
        # 0.0.0.0, exposes every session and every tool this process is
        # configured to run to whoever can reach the port. Warn loudly
        # instead of silently trusting the network it's bound to.
        print(
            f"[lulu-server] WARNING: binding to {args.host!r}, not localhost. "
            "This server has no authentication -- anyone who can reach this "
            "host/port can create sessions and run tools as this process. "
            "Put a real auth layer (or at least a reverse proxy with one) "
            "in front before binding beyond 127.0.0.1.",
            file=sys.stderr,
        )

    mode = AttentionMode(args.mode) if args.mode else None
    app = create_app(root=Path(args.root).resolve(), mode=mode)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    run()
