"""OpenRouterClient: OpenAICompatibleClient pointed at OpenRouter.

Not the platform this project is built on -- see docs/THESIS.md and the
README's architecture section, which spell out why ModelClient is a
Protocol in the first place: OpenRouter is the employer this project
targets, and being able to point Lulu at it with a ~20-line adapter (this
file) rather than a rewrite is itself part of the point.

Model choice matters here specifically: OpenRouter fronts hundreds of
models, and not all of them support tool/function calling. Pick one that
does (check the model's page on openrouter.ai for "Tools" support) --
OpenAICompatibleClient will raise ModelIncompatibleError with a clear
message if you pick one that doesn't, rather than failing silently or
with a cryptic provider error.
"""

from __future__ import annotations

import os

from lulu.llm.openai_compatible import DEFAULT_MAX_TOKENS, OpenAICompatibleClient

BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL = "anthropic/claude-sonnet-5"


class OpenRouterClient(OpenAICompatibleClient):
    name = "openrouter"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        fallback_models: list[str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client=None,
    ) -> None:
        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if resolved_key is None and client is None:
            raise ValueError(
                "OPENROUTER_API_KEY not set and no api_key/client given -- "
                "see .env.example"
            )
        super().__init__(
            base_url=BASE_URL,
            api_key=resolved_key,
            model=model,
            fallback_models=fallback_models,
            max_tokens=max_tokens,
            client=client,
        )
