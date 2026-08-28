"""AnthropicClient: message conversion, tool-call extraction, and --
the part worth the most test weight -- model fallback control flow. Every
test injects a FakeAnthropic double via the `client=` constructor param,
so none of this touches the network or needs an API key.
"""

from __future__ import annotations

import anthropic
import pytest

from lulu.llm.anthropic_client import AnthropicClient, _to_anthropic_messages
from lulu.llm.client import Message, ModelUnavailableError, TextDelta, ToolCall, ToolCallDelta, ToolResult, ToolSpec, UsageDelta
from .fakes.anthropic_sdk import (
    FakeAnthropic,
    FakeDelta,
    FakeResponse,
    FakeStreamContext,
    FakeStreamEvent,
    FakeTextBlock,
    FakeToolUseBlock,
    FakeUsage,
    connection_error,
    status_error,
)


def _client(fake: FakeAnthropic, **kwargs) -> AnthropicClient:
    return AnthropicClient(model="primary-model", fallback_models=["fallback-model"], client=fake, **kwargs)


def test_complete_returns_text_response():
    fake = FakeAnthropic()
    fake.messages.queue_create(FakeResponse(content=[FakeTextBlock("hello there")], usage=FakeUsage(12, 7)))
    client = _client(fake)

    response = client.complete([Message(role="user", content="hi")], tools=[])

    assert response.message.content == "hello there"
    assert response.message.tool_calls == []
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 7
    assert response.model == "primary-model"
    assert response.stop_reason == "end_turn"


def test_complete_extracts_tool_calls():
    fake = FakeAnthropic()
    fake.messages.queue_create(
        FakeResponse(content=[FakeToolUseBlock(id="tc_1", name="read_file", input={"path": "a.py"})])
    )
    client = _client(fake)

    response = client.complete([Message(role="user", content="read a.py")], tools=[])

    assert len(response.message.tool_calls) == 1
    tc = response.message.tool_calls[0]
    assert tc.id == "tc_1"
    assert tc.name == "read_file"
    assert tc.arguments == {"path": "a.py"}


def test_complete_falls_back_on_transient_error():
    fake = FakeAnthropic()
    fake.messages.queue_create(status_error(anthropic.OverloadedError, "busy", 529))
    fake.messages.queue_create(FakeResponse(content=[FakeTextBlock("ok from fallback")]))
    client = _client(fake)

    response = client.complete([Message(role="user", content="hi")], tools=[])

    assert response.model == "fallback-model"
    assert response.message.content == "ok from fallback"
    assert len(fake.messages.create_calls) == 2


def test_complete_raises_when_all_models_exhausted():
    fake = FakeAnthropic()
    original = status_error(anthropic.ServiceUnavailableError, "down", 503)
    fake.messages.queue_create(status_error(anthropic.OverloadedError, "busy", 529))
    fake.messages.queue_create(original)
    client = _client(fake)

    with pytest.raises(ModelUnavailableError) as exc_info:
        client.complete([Message(role="user", content="hi")], tools=[])

    assert exc_info.value.__cause__ is original
    assert len(fake.messages.create_calls) == 2


def test_complete_does_not_fall_back_on_auth_error():
    """A bad API key or malformed request isn't fixed by trying a
    different model -- it should propagate immediately, and fallback
    should never even be attempted."""
    fake = FakeAnthropic()
    fake.messages.queue_create(status_error(anthropic.AuthenticationError, "bad key", 401))
    client = _client(fake)

    with pytest.raises(anthropic.AuthenticationError):
        client.complete([Message(role="user", content="hi")], tools=[])

    assert len(fake.messages.create_calls) == 1


def test_complete_passes_not_given_when_no_tools():
    fake = FakeAnthropic()
    fake.messages.queue_create(FakeResponse(content=[FakeTextBlock("ok")]))
    client = _client(fake)

    client.complete([Message(role="user", content="hi")], tools=[])

    assert fake.messages.create_calls[0]["tools"] is anthropic.NOT_GIVEN


def test_complete_passes_tool_specs_when_present():
    fake = FakeAnthropic()
    fake.messages.queue_create(FakeResponse(content=[FakeTextBlock("ok")]))
    client = _client(fake)
    spec = ToolSpec(name="read_file", description="reads a file", input_schema={"type": "object"})

    client.complete([Message(role="user", content="hi")], tools=[spec])

    sent_tools = fake.messages.create_calls[0]["tools"]
    assert sent_tools == [{"name": "read_file", "description": "reads a file", "input_schema": {"type": "object"}}]


def test_message_conversion_includes_tool_results():
    messages = [
        Message(role="assistant", content="", tool_calls=[ToolCall(id="tc_1", name="read_file", arguments={"path": "a.py"})]),
        Message(role="user", tool_results=[ToolResult(tool_call_id="tc_1", content="file contents", is_error=False)]),
    ]

    converted = _to_anthropic_messages(messages)

    assert converted[0]["content"] == [
        {"type": "tool_use", "id": "tc_1", "name": "read_file", "input": {"path": "a.py"}}
    ]
    assert converted[1]["content"] == [
        {"type": "tool_result", "tool_use_id": "tc_1", "content": "file contents", "is_error": False}
    ]


def test_stream_yields_text_then_toolcall_then_usage():
    fake = FakeAnthropic()
    events = [
        FakeStreamEvent(type="message_start"),
        FakeStreamEvent(type="content_block_delta", delta=FakeDelta(text="Hel")),
        FakeStreamEvent(type="content_block_delta", delta=FakeDelta(text="lo")),
    ]
    final = FakeResponse(
        content=[FakeTextBlock("Hello"), FakeToolUseBlock(id="tc_1", name="bash", input={"command": "ls"})],
        usage=FakeUsage(20, 3),
        stop_reason="tool_use",
    )
    fake.messages.queue_stream(FakeStreamContext(events, final))
    client = _client(fake)

    collected = list(client.stream([Message(role="user", content="hi")], tools=[]))

    assert collected[0] == TextDelta(text="Hel")
    assert collected[1] == TextDelta(text="lo")
    assert isinstance(collected[2], ToolCallDelta)
    assert collected[2].tool_call == ToolCall(id="tc_1", name="bash", arguments={"command": "ls"})
    assert isinstance(collected[3], UsageDelta)
    assert collected[3].usage.input_tokens == 20
    assert collected[3].stop_reason == "tool_use"
    assert collected[3].model == "primary-model"


def test_stream_falls_back_on_transient_error():
    fake = FakeAnthropic()
    fake.messages.queue_stream(connection_error())
    final = FakeResponse(content=[FakeTextBlock("recovered")])
    fake.messages.queue_stream(FakeStreamContext([], final))
    client = _client(fake)

    collected = list(client.stream([Message(role="user", content="hi")], tools=[]))

    usage_event = collected[-1]
    assert isinstance(usage_event, UsageDelta)
    assert usage_event.model == "fallback-model"
    assert len(fake.messages.stream_calls) == 2


def test_stream_raises_when_all_models_exhausted():
    fake = FakeAnthropic()
    fake.messages.queue_stream(status_error(anthropic.OverloadedError, "busy", 529))
    fake.messages.queue_stream(status_error(anthropic.OverloadedError, "still busy", 529))
    client = _client(fake)

    with pytest.raises(ModelUnavailableError):
        list(client.stream([Message(role="user", content="hi")], tools=[]))
