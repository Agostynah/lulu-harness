"""ModelClient: the harness's LLM abstraction.

OpenRouter is the eventual employer this project targets, not the platform
it's built on -- nothing in the harness is written against a single
provider's API. ModelClient is a thin Protocol; llm/anthropic_client.py is
the default adapter (Anthropic, since that's what runs this project day to
day), and OpenRouter/Ollama adapters (roadmap) are meant to be ~30-line
implementations of the same Protocol, not a rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Protocol, Union


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass
class Message:
    role: Literal["user", "assistant"]
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ModelResponse:
    message: Message
    usage: Usage
    stop_reason: str
    model: str


@dataclass
class TextDelta:
    text: str


@dataclass
class ToolCallDelta:
    tool_call: ToolCall


@dataclass
class UsageDelta:
    """Always the last event a stream() call yields -- the point at which
    usage/stop_reason/model are actually known. Kept as an explicit event
    rather than client instance state so a caller can fully determine cost
    from the event stream alone, with nothing implicit to get out of sync."""

    usage: Usage
    stop_reason: str
    model: str


StreamEvent = Union[TextDelta, ToolCallDelta, UsageDelta]


class ModelUnavailableError(Exception):
    """Every model in the client's fallback chain failed with a transient
    error. Distinct from a provider auth/request error, which propagates
    immediately instead of triggering fallback -- retrying a bad API key
    or malformed request against a different model wouldn't fix anything."""


class ModelClient(Protocol):
    """What any model backend must provide. The loop, tools, and UI never
    import a concrete provider -- only this Protocol."""

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str = "",
    ) -> ModelResponse: ...

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str = "",
    ) -> Iterator[StreamEvent]: ...
