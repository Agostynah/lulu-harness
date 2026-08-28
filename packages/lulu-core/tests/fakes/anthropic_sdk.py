"""A hand-rolled double for the bits of the `anthropic` SDK AnthropicClient
actually touches -- .messages.create(), .messages.stream(), and the shape
of what comes back. Lets tests drive fallback/error paths deterministically
with no network and no API key, which is exactly the coverage that matters
since none of lulu.llm.anthropic_client's logic is provider-specific --
it's message conversion and fallback control flow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import anthropic
import httpx2

_FAKE_REQUEST = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(cls: type[Exception], message: str, status_code: int = 500) -> Exception:
    """Builds a real anthropic.APIStatusError subclass instance -- these
    require a genuine httpx2.Response (they read .request off it in
    __init__), so a plain `cls(message, response=None, body=None)` blows
    up with an AttributeError before the test even gets to exercise
    AnthropicClient's fallback logic."""
    response = httpx2.Response(status_code, request=_FAKE_REQUEST)
    return cls(message, response=response, body=None)


def connection_error(message: str = "connection failed") -> anthropic.APIConnectionError:
    return anthropic.APIConnectionError(message=message, request=_FAKE_REQUEST)


@dataclass
class FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class FakeToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class FakeUsage:
    input_tokens: int = 10
    output_tokens: int = 5


@dataclass
class FakeResponse:
    content: list[Any]
    usage: FakeUsage = field(default_factory=FakeUsage)
    stop_reason: str = "end_turn"


@dataclass
class FakeDelta:
    text: str
    type: str = "text_delta"


@dataclass
class FakeStreamEvent:
    type: str
    delta: FakeDelta | None = None


class FakeStreamContext:
    """Mimics `with client.messages.stream(...) as stream:` -- iterating
    yields the scripted events, get_final_message() returns the scripted
    final response, exactly like the real SDK's stream manager."""

    def __init__(self, events: list[FakeStreamEvent], final: FakeResponse):
        self._events = events
        self._final = final

    def __enter__(self) -> "FakeStreamContext":
        return self

    def __exit__(self, *exc_info: Any) -> bool:
        return False

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self) -> FakeResponse:
        return self._final


class FakeMessagesAPI:
    """Queue-based double: each call to create()/stream() pops the next
    queued result (a FakeResponse/FakeStreamContext to return, or an
    Exception instance to raise), so a test can script exactly the
    fallback sequence it wants to exercise."""

    def __init__(self) -> None:
        self._create_queue: list[Any] = []
        self._stream_queue: list[Any] = []
        self.create_calls: list[dict[str, Any]] = []
        self.stream_calls: list[dict[str, Any]] = []

    def queue_create(self, result: Any) -> None:
        self._create_queue.append(result)

    def queue_stream(self, result: Any) -> None:
        self._stream_queue.append(result)

    def create(self, **kwargs: Any) -> FakeResponse:
        self.create_calls.append(kwargs)
        result = self._create_queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    def stream(self, **kwargs: Any) -> FakeStreamContext:
        self.stream_calls.append(kwargs)
        result = self._stream_queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeAnthropic:
    def __init__(self) -> None:
        self.messages = FakeMessagesAPI()
