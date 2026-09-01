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

Prompt caching: `system` and the last tool spec each get an ephemeral
cache breakpoint (see `_to_anthropic_system_blocks`/`_to_anthropic_tools`)
-- `context` (per-turn memory-router results) deliberately does not, so a
changing memory block every turn never invalidates the cached prefix.
Two breakpoints today (tools+system); a third, judge-gated one for a
consolidated memory block is a separate, not-yet-built experiment -- see
decisions_todo.md.
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
    # Tool definitions are stable for a whole session (Lulu's tool surface
    # is small and fixed -- read/write/edit/glob/grep/bash), so the last
    # one gets a cache breakpoint: Anthropic caches everything up to and
    # including a marked block, so this reuses the entire tools array
    # across every call in the session instead of reprocessing it.
    out = [
        {"name": t.name, "description": t.description, "input_schema": t.input_schema}
        for t in tools
    ]
    if out:
        out[-1] = {**out[-1], "cache_control": {"type": "ephemeral"}}
    return out


def _to_anthropic_system_blocks(system: str, context: str) -> list[dict[str, Any]] | Any:
    """Builds the `system` argument sent to the SDK as content blocks, not
    a plain string, so a cache breakpoint can be placed after `system` but
    before `context` -- `system` is stable across calls in a session and
    worth caching; `context` (memory-router results) changes every turn
    and must stay outside the cached prefix, or every turn's different
    memory content would break the cache instead of reusing it."""
    blocks: list[dict[str, Any]] = []
    if system:
        blocks.append({"type": "text", "text": system, "cache_control": {"type": "ephemeral"}})
    if context:
        blocks.append({"type": "text", "text": context})
    return blocks if blocks else anthropic.NOT_GIVEN


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
        context: str = "",
    ) -> ModelResponse:
        last_error: Exception | None = None
        for model in self._model_chain():
            try:
                response = self._client.messages.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    system=_to_anthropic_system_blocks(system, context),
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
        context: str = "",
    ) -> Iterator[StreamEvent]:
        last_error: Exception | None = None
        for model in self._model_chain():
            try:
                with self._client.messages.stream(
                    model=model,
                    max_tokens=self.max_tokens,
                    system=_to_anthropic_system_blocks(system, context),
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
