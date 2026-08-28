"""OpenAICompatibleClient: message conversion (including the real format
difference from Anthropic -- each tool result is its own "tool"-role
message, not bundled into one block list), fallback control flow, and the
agentic-capability requirement (ModelIncompatibleError on a tool-related
BadRequestError). Same fake-injection pattern as test_anthropic_client.py
-- no network, no key needed.
"""

from __future__ import annotations

import openai
import pytest

from lulu.llm.client import Message, ModelUnavailableError, TextDelta, ToolCall, ToolCallDelta, ToolResult, ToolSpec, UsageDelta
from lulu.llm.openai_compatible import ModelIncompatibleError, OpenAICompatibleClient, _to_openai_messages

from .fakes.openai_sdk import (
    FakeChatCompletion,
    FakeChoice,
    FakeFunction,
    FakeMessage,
    FakeOpenAI,
    FakeToolCall,
    FakeUsage,
    connection_error,
    status_error,
)


def _client(fake: FakeOpenAI, **kwargs) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        base_url="https://example.com/v1",
        api_key="test",
        model="primary-model",
        fallback_models=["fallback-model"],
        client=fake,
        **kwargs,
    )


def test_complete_returns_text_response():
    fake = FakeOpenAI()
    fake.chat.completions.queue(FakeChatCompletion(choices=[FakeChoice(FakeMessage(content="hi there"))], usage=FakeUsage(12, 7)))
    client = _client(fake)

    response = client.complete([Message(role="user", content="hi")], tools=[])

    assert response.message.content == "hi there"
    assert response.usage.input_tokens == 12
    assert response.usage.output_tokens == 7
    assert response.model == "primary-model"
    assert response.stop_reason == "stop"


def test_complete_extracts_tool_calls_and_parses_json_arguments():
    fake = FakeOpenAI()
    fake.chat.completions.queue(
        FakeChatCompletion(
            choices=[
                FakeChoice(
                    FakeMessage(
                        tool_calls=[FakeToolCall(id="tc_1", function=FakeFunction(name="read_file", arguments='{"path": "a.py"}'))]
                    )
                )
            ]
        )
    )
    client = _client(fake)

    response = client.complete([Message(role="user", content="read a.py")], tools=[])

    tc = response.message.tool_calls[0]
    assert tc.id == "tc_1"
    assert tc.name == "read_file"
    assert tc.arguments == {"path": "a.py"}


def test_complete_falls_back_on_transient_error():
    fake = FakeOpenAI()
    fake.chat.completions.queue(status_error(openai.InternalServerError, "down", 500))
    fake.chat.completions.queue(FakeChatCompletion(choices=[FakeChoice(FakeMessage(content="ok"))]))
    client = _client(fake)

    response = client.complete([Message(role="user", content="hi")], tools=[])

    assert response.model == "fallback-model"
    assert response.message.content == "ok"


def test_complete_raises_when_all_models_exhausted():
    fake = FakeOpenAI()
    original = status_error(openai.InternalServerError, "still down", 500)
    fake.chat.completions.queue(status_error(openai.RateLimitError, "rate limited", 429))
    fake.chat.completions.queue(original)
    client = _client(fake)

    with pytest.raises(ModelUnavailableError) as exc_info:
        client.complete([Message(role="user", content="hi")], tools=[])

    assert exc_info.value.__cause__ is original


def test_complete_does_not_fall_back_on_auth_error():
    fake = FakeOpenAI()
    fake.chat.completions.queue(status_error(openai.AuthenticationError, "bad key", 401))
    client = _client(fake)

    with pytest.raises(openai.AuthenticationError):
        client.complete([Message(role="user", content="hi")], tools=[])

    assert len(fake.chat.completions.calls) == 1


def test_tool_incompatible_model_raises_clear_error():
    fake = FakeOpenAI()
    fake.chat.completions.queue(
        status_error(openai.BadRequestError, "this model does not support the 'tools' parameter", 400)
    )
    client = _client(fake)
    spec = ToolSpec(name="read_file", description="reads", input_schema={"type": "object"})

    with pytest.raises(ModelIncompatibleError, match="agentic"):
        client.complete([Message(role="user", content="hi")], tools=[spec])


def test_unrelated_bad_request_is_not_reinterpreted_as_incompatible():
    """A 400 that has nothing to do with tool support should propagate as
    a normal BadRequestError, not get misdiagnosed."""
    fake = FakeOpenAI()
    fake.chat.completions.queue(status_error(openai.BadRequestError, "invalid model id", 400))
    client = _client(fake)

    with pytest.raises(openai.BadRequestError):
        client.complete([Message(role="user", content="hi")], tools=[])


def test_bad_request_without_tools_in_the_call_is_not_reinterpreted():
    """Even if the message happens to mention 'tool', it shouldn't be
    treated as an incompatibility unless tools were actually requested."""
    fake = FakeOpenAI()
    fake.chat.completions.queue(status_error(openai.BadRequestError, "no tool talk here, just a bad prompt", 400))
    client = _client(fake)

    with pytest.raises(openai.BadRequestError):
        client.complete([Message(role="user", content="hi")], tools=[])


def test_message_conversion_prepends_system_message():
    converted = _to_openai_messages([Message(role="user", content="hi")], system="be nice")
    assert converted[0] == {"role": "system", "content": "be nice"}


def test_message_conversion_expands_tool_results_into_separate_messages():
    """Real format difference from Anthropic: OpenAI wants one message
    per tool result, not one message bundling several."""
    messages = [
        Message(
            role="user",
            tool_results=[
                ToolResult(tool_call_id="tc_1", content="result A"),
                ToolResult(tool_call_id="tc_2", content="result B"),
            ],
        )
    ]

    converted = _to_openai_messages(messages, system="")

    assert converted == [
        {"role": "tool", "tool_call_id": "tc_1", "content": "result A"},
        {"role": "tool", "tool_call_id": "tc_2", "content": "result B"},
    ]


def test_message_conversion_serializes_tool_call_arguments_as_json_string():
    messages = [
        Message(role="assistant", tool_calls=[ToolCall(id="tc_1", name="read_file", arguments={"path": "a.py"})])
    ]

    converted = _to_openai_messages(messages, system="")

    assert converted[0]["tool_calls"][0]["function"]["arguments"] == '{"path": "a.py"}'


def test_connection_error_triggers_fallback():
    fake = FakeOpenAI()
    fake.chat.completions.queue(connection_error())
    fake.chat.completions.queue(FakeChatCompletion(choices=[FakeChoice(FakeMessage(content="recovered"))]))
    client = _client(fake)

    response = client.complete([Message(role="user", content="hi")], tools=[])

    assert response.model == "fallback-model"


def test_stream_reuses_complete_and_emits_same_shape_of_events():
    fake = FakeOpenAI()
    fake.chat.completions.queue(
        FakeChatCompletion(
            choices=[
                FakeChoice(
                    FakeMessage(
                        content="hi",
                        tool_calls=[FakeToolCall(id="tc_1", function=FakeFunction(name="bash", arguments="{}"))],
                    )
                )
            ],
            usage=FakeUsage(20, 3),
        )
    )
    client = _client(fake)

    events = list(client.stream([Message(role="user", content="hi")], tools=[]))

    assert events[0] == TextDelta(text="hi")
    assert isinstance(events[1], ToolCallDelta)
    assert events[1].tool_call.name == "bash"
    assert isinstance(events[2], UsageDelta)
    assert events[2].usage.input_tokens == 20
