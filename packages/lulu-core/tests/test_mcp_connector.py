"""McpShardStore: search_async's logic against a fake ClientSession (no
real transport, no real server -- see decisions_todo.md for the open item
about which real MCP server to point this at for a live smoke test), plus
the sync search() bridge and latency measurement. Async methods are run
directly via asyncio.run() inside plain sync test functions -- no need
for a pytest-asyncio dependency for coverage this small.
"""

from __future__ import annotations

import asyncio

import pytest

from lulu.connectors.mcp import McpShardStore, McpToolError
from .fakes.mcp_sdk import FakeCallToolResult, FakeClientSession, FakeTextContent


def test_search_async_returns_results_from_text_content():
    session = FakeClientSession()
    session.queue(
        FakeCallToolResult(
            content=[FakeTextContent("first result"), FakeTextContent("second result")]
        )
    )
    store = McpShardStore(session=session, tool_name="search_issues")

    results = asyncio.run(store.search_async(k=5, query="auth bug"))

    assert [r.content for r in results] == ["first result", "second result"]
    assert results[0].score > results[1].score  # earlier results rank higher


def test_search_async_passes_query_and_limit_as_arguments():
    session = FakeClientSession()
    session.queue(FakeCallToolResult(content=[]))
    store = McpShardStore(session=session, tool_name="search_issues", query_arg="q", result_limit_arg="max")

    asyncio.run(store.search_async(k=7, query="auth bug"))

    tool_name, arguments = session.calls[0]
    assert tool_name == "search_issues"
    assert arguments == {"q": "auth bug", "max": 7}


def test_search_async_omits_limit_arg_when_none():
    session = FakeClientSession()
    session.queue(FakeCallToolResult(content=[]))
    store = McpShardStore(session=session, tool_name="search_issues", result_limit_arg=None)

    asyncio.run(store.search_async(k=7, query="x"))

    _, arguments = session.calls[0]
    assert "limit" not in arguments


def test_search_async_respects_k_even_if_server_returns_more():
    session = FakeClientSession()
    session.queue(
        FakeCallToolResult(content=[FakeTextContent(f"result {i}") for i in range(10)])
    )
    store = McpShardStore(session=session, tool_name="search_issues")

    results = asyncio.run(store.search_async(k=3, query="x"))

    assert len(results) == 3


def test_search_async_skips_non_text_content_blocks():
    class FakeImageContent:
        type = "image"

    session = FakeClientSession()
    session.queue(FakeCallToolResult(content=[FakeImageContent(), FakeTextContent("only this one")]))
    store = McpShardStore(session=session, tool_name="search_issues")

    results = asyncio.run(store.search_async(k=5, query="x"))

    assert len(results) == 1
    assert results[0].content == "only this one"


def test_search_async_raises_mcp_tool_error_on_is_error_result():
    session = FakeClientSession()
    session.queue(FakeCallToolResult(content=[FakeTextContent("rate limited")], is_error=True))
    store = McpShardStore(session=session, tool_name="search_issues")

    with pytest.raises(McpToolError, match="rate limited"):
        asyncio.run(store.search_async(k=5, query="x"))


def test_search_sync_bridge_returns_same_results():
    session = FakeClientSession()
    session.queue(FakeCallToolResult(content=[FakeTextContent("bridged result")]))
    store = McpShardStore(session=session, tool_name="search_issues")

    results = store.search(query_vec=None, k=5, query="x")

    assert results[0].content == "bridged result"


def test_len_is_zero_meaning_unknown_not_empty():
    store = McpShardStore(session=FakeClientSession(), tool_name="search_issues")
    assert len(store) == 0


def test_measured_latency_is_none_before_any_call():
    store = McpShardStore(session=FakeClientSession(), tool_name="search_issues")
    assert store.measured_latency_ms() is None


def test_measured_latency_is_recorded_after_calls():
    session = FakeClientSession()
    session.queue(FakeCallToolResult(content=[]))
    session.queue(FakeCallToolResult(content=[]))
    store = McpShardStore(session=session, tool_name="search_issues")

    asyncio.run(store.search_async(k=5, query="a"))
    asyncio.run(store.search_async(k=5, query="b"))

    latency = store.measured_latency_ms()
    assert latency is not None
    assert latency >= 0
    assert len(store._latencies_ms) == 2
