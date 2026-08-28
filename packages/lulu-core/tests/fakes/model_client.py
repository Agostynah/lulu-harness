"""A scripted ModelClient double for testing loop.py without touching any
real provider. Each call to complete() pops the next queued ModelResponse
(or raises if the queue is empty, which is itself a useful test failure --
it means the loop called the model more times than the test expected)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lulu.llm.client import Message, ModelResponse, ToolSpec, Usage


@dataclass
class FakeModelClient:
    responses: list[ModelResponse] = field(default_factory=list)
    calls: list[tuple[list[Message], list[ToolSpec], str]] = field(default_factory=list)

    def queue(self, response: ModelResponse) -> None:
        self.responses.append(response)

    def complete(self, messages: list[Message], tools: list[ToolSpec], system: str = "") -> ModelResponse:
        self.calls.append((list(messages), list(tools), system))
        if not self.responses:
            raise AssertionError("FakeModelClient.complete() called with no queued response left")
        return self.responses.pop(0)

    def stream(self, messages, tools, system=""):
        raise NotImplementedError("FakeModelClient is complete()-only; loop.py v0 doesn't stream")


def text_response(text: str, model: str = "fake-model") -> ModelResponse:
    return ModelResponse(
        message=Message(role="assistant", content=text),
        usage=Usage(input_tokens=10, output_tokens=5),
        stop_reason="end_turn",
        model=model,
    )


def tool_call_response(tool_calls, model: str = "fake-model") -> ModelResponse:
    return ModelResponse(
        message=Message(role="assistant", content="", tool_calls=list(tool_calls)),
        usage=Usage(input_tokens=10, output_tokens=5),
        stop_reason="tool_use",
        model=model,
    )
