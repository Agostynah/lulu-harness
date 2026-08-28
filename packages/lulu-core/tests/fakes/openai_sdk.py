"""A hand-rolled double for the bits of the `openai` SDK
OpenAICompatibleClient actually touches -- .chat.completions.create() and
the shape of what comes back. Same pattern as fakes/anthropic_sdk.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx2
import openai

_FAKE_REQUEST = httpx2.Request("POST", "https://example.com/v1/chat/completions")


def status_error(cls: type[Exception], message: str, status_code: int = 500) -> Exception:
    """openai's APIStatusError subclasses need a real httpx2.Response in
    their constructor (they read .request off it), same gotcha as the
    anthropic SDK -- see tests/fakes/anthropic_sdk.py."""
    response = httpx2.Response(status_code, request=_FAKE_REQUEST)
    return cls(message, response=response, body=None)


def connection_error(message: str = "connection failed") -> openai.APIConnectionError:
    return openai.APIConnectionError(message=message, request=_FAKE_REQUEST)


@dataclass
class FakeFunction:
    name: str
    arguments: str  # JSON-encoded string, matching the real SDK's shape


@dataclass
class FakeToolCall:
    id: str
    function: FakeFunction
    type: str = "function"


@dataclass
class FakeMessage:
    content: str | None = None
    tool_calls: list[FakeToolCall] | None = None


@dataclass
class FakeChoice:
    message: FakeMessage
    finish_reason: str = "stop"
    index: int = 0


@dataclass
class FakeUsage:
    prompt_tokens: int = 10
    completion_tokens: int = 5


@dataclass
class FakeChatCompletion:
    choices: list[FakeChoice]
    usage: FakeUsage = field(default_factory=FakeUsage)


class FakeCompletionsAPI:
    def __init__(self) -> None:
        self._queue: list[Any] = []
        self.calls: list[dict[str, Any]] = []

    def queue(self, result: Any) -> None:
        self._queue.append(result)

    def create(self, **kwargs: Any) -> FakeChatCompletion:
        self.calls.append(kwargs)
        result = self._queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeChat:
    def __init__(self) -> None:
        self.completions = FakeCompletionsAPI()


class FakeOpenAI:
    def __init__(self) -> None:
        self.chat = FakeChat()
