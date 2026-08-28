"""server.py: the FastAPI surface over the same wiring cli.py uses,
verified via FastAPI's TestClient (runs the ASGI app in-process -- no
real network, no browser needed, which matters this session since no
browser tool is available to verify the frontend visually). The last two
tests are the ones that actually matter: memory retrieval produces a real
trace through the HTTP layer, and server_ask_human's fail-closed design
means a blast-radius command attempted through the API gets refused, not
silently executed and not silently hung.
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from lulu.attention import AttentionMode
from lulu.llm.client import ToolCall
from lulu.memory import MemoryStore
from lulu.server import create_app

from .fakes.embedder import FakeEmbedder, ZeroEmbedder
from .fakes.model_client import FakeModelClient, text_response, tool_call_response


def _client(tmp_path: Path, model=None, mode=None, memory=None) -> TestClient:
    root = tmp_path / "proj"
    root.mkdir(exist_ok=True)
    app = create_app(
        root=root,
        model_override=model or FakeModelClient(),
        memory_override=memory or MemoryStore(embedder=ZeroEmbedder()),
        mode=mode,
    )
    return TestClient(app)


def test_health():
    client = _client(Path("."))
    assert client.get("/api/health").json() == {"status": "ok"}


def test_config_reflects_default_server_mode(tmp_path: Path):
    client = _client(tmp_path)
    config = client.get("/api/config").json()
    assert config["attention_mode"] == "auto_edits"


def test_config_reflects_explicit_mode_override(tmp_path: Path):
    client = _client(tmp_path, mode=AttentionMode.PLAN)
    config = client.get("/api/config").json()
    assert config["attention_mode"] == "plan"


def test_create_session_returns_a_new_id(tmp_path: Path):
    client = _client(tmp_path)
    response = client.post("/api/sessions")
    body = response.json()
    assert response.status_code == 200
    assert body["session_id"]
    assert body["history"] == []


def test_create_session_resumes_an_existing_id(tmp_path: Path):
    client = _client(tmp_path)
    first = client.post("/api/sessions").json()

    resumed = client.post("/api/sessions", params={"session_id": first["session_id"]}).json()

    assert resumed["session_id"] == first["session_id"]


def test_get_history_for_unknown_session_404s(tmp_path: Path):
    client = _client(tmp_path)
    response = client.get("/api/sessions/does-not-exist/history")
    assert response.status_code == 404


def test_run_turn_returns_final_text_and_usage(tmp_path: Path):
    model = FakeModelClient()
    model.queue(text_response("hello back"))
    client = _client(tmp_path, model=model)
    session_id = client.post("/api/sessions").json()["session_id"]

    response = client.post(f"/api/sessions/{session_id}/turn", json={"prompt": "hi"})
    body = response.json()

    assert response.status_code == 200
    assert body["final_text"] == "hello back"
    assert body["stopped_reason"] == "final_text"
    assert body["usage_totals"]["input_tokens"] == 10


def test_run_turn_persists_to_session_history(tmp_path: Path):
    model = FakeModelClient()
    model.queue(text_response("hello back"))
    client = _client(tmp_path, model=model)
    session_id = client.post("/api/sessions").json()["session_id"]

    client.post(f"/api/sessions/{session_id}/turn", json={"prompt": "hi"})
    history = client.get(f"/api/sessions/{session_id}/history").json()["history"]

    assert history[0]["content"] == "hi"
    assert history[-1]["content"] == "hello back"


def test_run_turn_with_memory_produces_a_real_trace(tmp_path: Path):
    embedder = FakeEmbedder()
    embedder.register("decided to use SQLite", [1.0, 0.0, 0.0])
    embedder.register("what did we decide", [1.0, 0.0, 0.0])
    embedder.register("User: what did we decide\nLulu: SQLite", [0.9, 0.1, 0.0])
    memory = MemoryStore(embedder=embedder, strategy="query_all", k=5)
    memory.write("decided to use SQLite", shard="episodic")

    model = FakeModelClient()
    model.queue(text_response("SQLite"))
    client = _client(tmp_path, model=model, memory=memory)
    session_id = client.post("/api/sessions").json()["session_id"]

    body = client.post(f"/api/sessions/{session_id}/turn", json={"prompt": "what did we decide"}).json()

    assert body["trace"] is not None
    assert len(body["trace"]["results"]) > 0
    assert body["trace"]["results"][0]["content"] == "decided to use SQLite"


def test_stream_endpoint_emits_sse_events(tmp_path: Path):
    model = FakeModelClient()
    model.queue(text_response("streamed reply"))
    client = _client(tmp_path, model=model)
    session_id = client.post("/api/sessions").json()["session_id"]

    with client.stream(
        "GET", f"/api/sessions/{session_id}/turn/stream", params={"prompt": "hi"}
    ) as response:
        body = "".join(response.iter_text())

    assert "event: message" in body
    assert "streamed reply" in body
    assert "event: done" in body


def test_blast_radius_command_via_server_is_denied_not_executed(tmp_path: Path):
    """The security-relevant one: server_ask_human always denies (see
    module docstring), so a bash command that would normally need human
    confirmation -- even under AUTO mode, which would otherwise ALLOW
    ordinary bash calls with zero confirmation -- gets refused through
    the HTTP layer, not silently run and not silently hung forever."""
    model = FakeModelClient()
    model.queue(
        tool_call_response([ToolCall(id="tc1", name="bash", arguments={"command": "rm -rf /"})])
    )
    model.queue(text_response("could not do that"))
    client = _client(tmp_path, model=model, mode=AttentionMode.AUTO)
    session_id = client.post("/api/sessions").json()["session_id"]

    client.post(f"/api/sessions/{session_id}/turn", json={"prompt": "delete everything"})
    history = client.get(f"/api/sessions/{session_id}/history").json()["history"]

    tool_result_message = next(m for m in history if m["tool_results"])
    assert tool_result_message["tool_results"][0]["is_error"] is True
    assert "denied" in tool_result_message["tool_results"][0]["content"]
