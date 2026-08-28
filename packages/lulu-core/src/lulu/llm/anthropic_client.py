"""AnthropicClient: the default ModelClient adapter.

Talks to the real `anthropic` SDK only inside this module -- every other
part of the harness imports lulu.llm.client's Protocol, never this class
directly, so swapping in OpenRouterClient/OllamaClient later touches this
file and nothing else.

Model fallback: a chain of models is tried in order. Transient
provider-side failures (overloaded, rate-limited, 5xx, connection/timeout)
advance to the next model in the chain. Anything else (bad API key,
malformed request) propagates immediately -- retrying against a different
model wouldn't fix either of those, and silently swallowing them would
hide a real bug.
"""

from __future__ import annotations

from typing import Any, Iterator

import anthropic

from lulu.llm.client import (
    Message,
    ModelResponse,
    ModelUnavailableError,
    StreamEvent,
    TextDelta,
    ToolCall,
    ToolCallDelta,
    ToolSpec,
    Usage,
    UsageDelta,
)

DEFAULT_MODEL = "claude-sonnet-5"
DEFAULT_FALLBACK_MODELS = ["claude-opus-5"]
DEFAULT_MAX_TOKENS = 4096

# Only these are worth retrying against a different model -- everything
# else (auth, bad request, permission) is a caller-fixable problem, not a
# provider outage.
TRANSIENT_ERRORS = (
    anthropic.OverloadedError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.ServiceUnavailableError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
)


def _to_anthropic_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        blocks: list[dict[str, Any]] = []
        if m.content:
            blocks.append({"type": "text", "text": m.content})
        for tc in m.tool_calls:
            blocks.append({"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.arguments})
        for tr in m.tool_results:
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tr.tool_call_id,
                    "content": tr.content,
                    "is_error": tr.is_error,
                }
            )
        out.append({"role": m.role, "content": blocks})
    return out


def _from_anthropic_content(content: list[Any]) -> tuple[str, list[ToolCall]]:
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for block in content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))
    return "".join(text_parts), tool_calls


class AnthropicClient:
    name = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        fallback_models: list[str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any = None,
    ) -> None:
        self.model = model
        self.fallback_models = fallback_models if fallback_models is not None else list(DEFAULT_FALLBACK_MODELS)
        self.max_tokens = max_tokens
        # `client` is injectable so tests never touch the real SDK/network.
        self._client = client if client is not None else anthropic.Anthropic(api_key=api_key)

    def _model_chain(self) -> list[str]:
        return [self.model, *self.fallback_models]

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str = "",
    ) -> ModelResponse:
        last_error: Exception | None = None
        for model in self._model_chain():
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    system=system or anthropic.NOT_GIVEN,
                    messages=_to_anthropic_messages(messages),
                    tools=_to_anthropic_tools(tools) if tools else anthropic.NOT_GIVEN,
                )
            except TRANSIENT_ERRORS as exc:
                last_error = exc
                continue

            text, tool_calls = _from_anthropic_content(response.content)
            return ModelResponse(
                message=Message(role="assistant", content=text, tool_calls=tool_calls),
                usage=Usage(
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                ),
                stop_reason=response.stop_reason,
                model=model,
            )

        raise ModelUnavailableError(
            f"all models exhausted: {self._model_chain()}"
        ) from last_error

    def stream(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str = "",
    ) -> Iterator[StreamEvent]:
        last_error: Exception | None = None
        for model in self._model_chain():
            try:
                with self._client.messages.stream(
                    model=model,
                    max_tokens=self.max_tokens,
                    system=system or anthropic.NOT_GIVEN,
                    messages=_to_anthropic_messages(messages),
                    tools=_to_anthropic_tools(tools) if tools else anthropic.NOT_GIVEN,
                ) as stream:
                    for event in stream:
                        if event.type == "content_block_delta" and event.delta.type == "text_delta":
                            yield TextDelta(text=event.delta.text)
                    final = stream.get_final_message()
            except TRANSIENT_ERRORS as exc:
                last_error = exc
                continue

            _text, tool_calls = _from_anthropic_content(final.content)
            for tc in tool_calls:
                yield ToolCallDelta(tool_call=tc)
            yield UsageDelta(
                usage=Usage(
                    input_tokens=final.usage.input_tokens,
                    output_tokens=final.usage.output_tokens,
                ),
                stop_reason=final.stop_reason,
                model=model,
            )
            return

        raise ModelUnavailableError(
            f"all models exhausted: {self._model_chain()}"
        ) from last_error
