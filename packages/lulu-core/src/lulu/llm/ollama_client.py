"""OllamaClient: OpenAICompatibleClient pointed at a local Ollama server.

No real API key needed -- Ollama's OpenAI-compatible endpoint doesn't
authenticate locally, it just requires the SDK's api_key field to be a
non-empty string, which "ollama" satisfies.

Same tool-calling requirement as every other adapter (see
openai_compatible.py): pick a model that actually supports it. Not every
locally-runnable model does -- check Ollama's model library for "tools"
support before assuming one works.
"""

from __future__ import annotations

from lulu.llm.openai_compatible import DEFAULT_MAX_TOKENS, OpenAICompatibleClient

BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "llama3.1"
PLACEHOLDER_API_KEY = "ollama"


class OllamaClient(OpenAICompatibleClient):
    name = "ollama"

    def __init__(
        self,
        base_url: str = BASE_URL,
        model: str = DEFAULT_MODEL,
        fallback_models: list[str] | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        client=None,
    ) -> None:
        super().__init__(
            base_url=base_url,
            api_key=PLACEHOLDER_API_KEY,
            model=model,
            fallback_models=fallback_models,
            max_tokens=max_tokens,
            client=client,
        )
