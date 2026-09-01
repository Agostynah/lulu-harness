"""OpenAICompatibleClient: shared base for any provider that speaks the
OpenAI chat-completions API. OpenRouter and Ollama both do, so one
implementation backs both adapters (openrouter_client.py and
ollama_client.py are each a ~20-line factory that just sets a base_url,
a default model, and where the API key comes from) -- proving "any model,
same API" is an actual property of this codebase, not just something the
README claims.

Requires agentic (tool-calling) support. This is not an optional
capability to check for -- Lulu's entire loop (loop.py) is built around
tool_use round-tripping between the model and the harness. A model that
can't emit structured tool calls cannot run this harness at all, the same
way a car without wheels can't be a car with an optional feature missing.
If the provider rejects a request specifically because it included the
`tools` parameter, this raises ModelIncompatibleError with an explicit,
actionable message instead of surfacing the provider's often-cryptic
underlying error and leaving the caller to guess why. (Detection is a
heuristic -- it looks for "tool" in the rejected request's error message,
since providers don't share one standard error code for "this model can't
call tools" -- not a claim that every incompatible model is caught this
way, only that the common case gets a clearer message than the raw
provider error.)
"""

from __future__ import annotations

import json
from typing import Any

import openai

from typing import Iterator

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

DEFAULT_MAX_TOKENS = 4096

TRANSIENT_ERRORS = (
    openai.RateLimitError,
    openai.InternalServerError,
    openai.APIConnectionError,
    openai.APITimeoutError,
)


class ModelIncompatibleError(Exception):
    """The configured model rejected a request specifically because it
    included tool definitions -- it likely doesn't support agentic
    (tool-calling) use, which Lulu requires, not an optional extra."""


def _to_openai_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t.name,
                "description": t.description,
                "parameters": t.input_schema,
            },
        }
        for t in tools
    ]


def _to_openai_messages(messages: list[Message], system: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    if system:
        out.append({"role": "system", "content": system})

    for m in messages:
        if m.tool_results:
            # Unlike Anthropic's block format, OpenAI represents each
            # tool result as its own separate message with role="tool".
            for tr in m.tool_results:
                out.append({"role": "tool", "tool_call_id": tr.tool_call_id, "content": tr.content})
            continue

        entry: dict[str, Any] = {"role": m.role, "content": m.content or None}
        if m.tool_calls:
            entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in m.tool_calls
            ]
        out.append(entry)

    return out


def _from_openai_message(message: Any) -> tuple[str, list[ToolCall]]:
    text = message.content or ""
    tool_calls: list[ToolCall] = []
    for tc in message.tool_calls or []:
        try:
            arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
        except json.JSONDecodeError:
            arguments = {}
        tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
    return text, tool_calls


def _looks_like_a_tool_support_rejection(exc: openai.BadRequestError) -> bool:
    message = str(exc).lower()
    return "tool" in message or "function" in message


class OpenAICompatibleClient:
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        fallback_models: list[str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client: Any = None,
    ) -> None:
        self.model = model
        self.fallback_models = fallback_models if fallback_models is not None else []
        self.max_tokens = max_tokens
        self._client = client if client is not None else openai.OpenAI(base_url=base_url, api_key=api_key)

    def _model_chain(self) -> list[str]:
        return [self.model, *self.fallback_models]

    def complete(
        self,
        messages: list[Message],
        tools: list[ToolSpec],
        system: str = "",
        context: str = "",
    ) -> ModelResponse:
        # No explicit prompt-caching control on this path (unlike
        # AnthropicClient) -- OpenAI-compatible providers vary in whether/
        # how they cache, and Ollama-served local models don't benefit
        # from provider-side caching at all. `context` is folded back into
        # `system` here, same as loop.py did before this split existed.
        combined_system = f"{system}\n\n{context}".strip() if context else system
        last_error: Exception | None = None
        for model in self._model_chain():
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    max_tokens=self.max_tokens,
                    messages=_to_openai_messages(messages, combined_system),
                    tools=_to_openai_tools(tools) if tools else openai.omit,
                )
            except openai.BadRequestError as exc:
                if tools and _looks_like_a_tool_support_rejection(exc):
                    raise ModelIncompatibleError(
                        f"model {model!r} rejected a tool-calling request -- Lulu requires "
                        "agentic (tool-calling) support, which this model may not have. "
                        f"Underlying error: {exc}"
                    ) from exc
                raise
            except TRANSIENT_ERRORS as exc:
                last_error = exc
                continue

            choice = response.choices[0]
            text, tool_calls = _from_openai_message(choice.message)
            return ModelResponse(
                message=Message(role="assistant", content=text, tool_calls=tool_calls),
                usage=Usage(
                    input_tokens=response.usage.prompt_tokens,
                    output_tokens=response.usage.completion_tokens,
                ),
                stop_reason=choice.finish_reason,
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
        """Satisfies the ModelClient Protocol, but is NOT real token-level
        streaming yet: it calls complete() (getting the fallback logic for
        free, no duplication) and re-emits the whole result as one
        TextDelta + ToolCallDeltas + a UsageDelta. Real incremental
        streaming for OpenAI-compatible providers means reconstructing
        tool calls from partial `arguments` JSON strings split across
        many chunks, which is meaningfully more involved than Anthropic's
        get_final_message()-based approach -- not worth building before
        anything actually consumes it (the harness's loop is
        complete()-only today; nothing streams until the UI/server
        exists). When that lands, this can be swapped for real streaming
        without changing the Protocol or any caller.
        """
        response = self.complete(messages, tools, system=system, context=context)
        if response.message.content:
            yield TextDelta(text=response.message.content)
        for tc in response.message.tool_calls:
            yield ToolCallDelta(tool_call=tc)
        yield UsageDelta(usage=response.usage, stop_reason=response.stop_reason, model=response.model)
