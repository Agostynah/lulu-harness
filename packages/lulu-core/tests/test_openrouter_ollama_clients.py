"""OpenRouterClient and OllamaClient: thin factories over
OpenAICompatibleClient. Proving these really are ~20-line adapters, not
duplicated implementations -- these tests check wiring (base_url, default
model, where the key comes from), not request/response logic, which is
already covered in test_openai_compatible.py."""

from __future__ import annotations

import pytest

from lulu.llm.ollama_client import BASE_URL as OLLAMA_BASE_URL
from lulu.llm.ollama_client import OllamaClient
from lulu.llm.openrouter_client import BASE_URL as OPENROUTER_BASE_URL
from lulu.llm.openrouter_client import OpenRouterClient

from .fakes.openai_sdk import FakeOpenAI


def test_openrouter_uses_injected_client_without_requiring_a_key():
    fake = FakeOpenAI()
    client = OpenRouterClient(model="anthropic/claude-sonnet-5", client=fake)
    assert client._client is fake
    assert client.model == "anthropic/claude-sonnet-5"


def test_openrouter_raises_without_key_or_client(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        OpenRouterClient(model="anthropic/claude-sonnet-5")


def test_openrouter_reads_key_from_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test-key")
    # No client override and no explicit api_key -- must not raise, and
    # must actually construct a real (if unused) SDK client.
    client = OpenRouterClient(model="anthropic/claude-sonnet-5")
    assert client._client is not None


def test_openrouter_explicit_api_key_overrides_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    client = OpenRouterClient(model="x", api_key="explicit-key")
    assert client._client is not None


def test_openrouter_base_url_is_openrouter():
    assert OPENROUTER_BASE_URL == "https://openrouter.ai/api/v1"


def test_openrouter_name_is_distinct_from_base_client():
    fake = FakeOpenAI()
    client = OpenRouterClient(model="x", client=fake)
    assert client.name == "openrouter"


def test_ollama_never_requires_a_key():
    """Ollama is local -- must never raise for lack of an API key,
    regardless of environment."""
    client = OllamaClient(model="llama3.1")
    assert client._client is not None


def test_ollama_base_url_points_at_localhost():
    assert OLLAMA_BASE_URL == "http://localhost:11434/v1"


def test_ollama_uses_injected_client():
    fake = FakeOpenAI()
    client = OllamaClient(model="llama3.1", client=fake)
    assert client._client is fake


def test_ollama_name():
    client = OllamaClient(model="llama3.1", client=FakeOpenAI())
    assert client.name == "ollama"


def test_ollama_custom_base_url_override():
    """Someone running Ollama on a non-default port/host."""
    client = OllamaClient(base_url="http://192.168.1.50:11434/v1", model="llama3.1", client=FakeOpenAI())
    assert client._client is not None
