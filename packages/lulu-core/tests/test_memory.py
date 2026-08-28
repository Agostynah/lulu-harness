"""MemoryStore: the piece that makes AgentLoop's memory retrieval real.
Uses FakeEmbedder throughout so similarity is fully controlled and no
real model ever loads."""

from __future__ import annotations

import pytest
from lulu_router.cost import Budget

from lulu.memory import MemoryStore

from .fakes.embedder import FakeEmbedder


def _store() -> tuple[MemoryStore, FakeEmbedder]:
    embedder = FakeEmbedder()
    return MemoryStore(embedder=embedder, strategy="query_all", k=5), embedder


def test_write_then_search_surfaces_the_memory():
    store, embedder = _store()
    embedder.register("decided to use SQLite for the cache", [1.0, 0.0, 0.0])
    embedder.register("what did we decide about the cache", [1.0, 0.0, 0.0])

    store.write("decided to use SQLite for the cache", shard="episodic")
    result = store.search("what did we decide about the cache")

    assert "decided to use SQLite for the cache" in result.text


def test_search_with_no_memories_returns_empty_text():
    store, embedder = _store()
    embedder.register("anything", [1.0, 0.0])
    result = store.search("anything")
    assert result.text == ""
    assert result.trace.results == []


def test_unknown_shard_raises_on_write():
    store, embedder = _store()
    embedder.register("x", [1.0, 0.0])
    with pytest.raises(ValueError, match="unknown shard"):
        store.write("x", shard="not_a_real_shard")


def test_write_updates_shard_centroid():
    store, embedder = _store()
    embedder.register("first memory", [1.0, 0.0, 0.0])

    store.write("first memory", shard="episodic")

    shard = store._shards[("episodic", None)]
    assert shard.centroid is not None
    assert pytest.approx(float((shard.centroid**2).sum()), abs=1e-5) == 1.0  # normalized


def test_multiple_writes_accumulate_in_the_same_shard():
    store, embedder = _store()
    embedder.register("memory one", [1.0, 0.0, 0.0])
    embedder.register("memory two", [0.9, 0.1, 0.0])
    embedder.register("query", [0.95, 0.05, 0.0])

    store.write("memory one", shard="episodic")
    store.write("memory two", shard="episodic")
    result = store.search("query")

    assert "memory one" in result.text
    assert "memory two" in result.text


def test_scoped_write_is_invisible_to_a_different_scope():
    store, embedder = _store()
    embedder.register("customer A's secret", [1.0, 0.0, 0.0])
    embedder.register("query", [1.0, 0.0, 0.0])

    store.write("customer A's secret", shard="episodic", scope="customer-a")
    result = store.search("query", scope="customer-b")

    assert "customer A's secret" not in result.text


def test_scoped_write_is_visible_to_the_matching_scope():
    store, embedder = _store()
    embedder.register("customer A's secret", [1.0, 0.0, 0.0])
    embedder.register("query", [1.0, 0.0, 0.0])

    store.write("customer A's secret", shard="episodic", scope="customer-a")
    result = store.search("query", scope="customer-a")

    assert "customer A's secret" in result.text


def test_scoped_write_is_invisible_to_an_unscoped_search():
    """Once a shard is scoped at all, an anonymous (scope=None) caller
    should not see it either -- scoping is an allowlist, not a denylist
    that defaults open."""
    store, embedder = _store()
    embedder.register("customer A's secret", [1.0, 0.0, 0.0])
    embedder.register("query", [1.0, 0.0, 0.0])

    store.write("customer A's secret", shard="episodic", scope="customer-a")
    result = store.search("query", scope=None)

    assert "customer A's secret" not in result.text


def test_two_different_scopes_writing_the_same_shard_type_stay_isolated():
    """The actual bug an adversarial review caught: an earlier
    implementation kept ONE physical shard per type and UNIONED scopes
    into its allowed_scopes on write, so once customer-a and customer-b
    both wrote to "episodic", shard.permits() passed for either scope and
    the merged store (both customers' content) was searchable by both.
    Two scopes writing to the same shard TYPE must produce two physically
    separate shards, never one merged store two scopes can both pass the
    permission check for."""
    store, embedder = _store()
    embedder.register("customer A's secret", [1.0, 0.0, 0.0])
    embedder.register("customer B's secret", [1.0, 0.0, 0.0])  # deliberately identical vector
    embedder.register("query", [1.0, 0.0, 0.0])

    store.write("customer A's secret", shard="episodic", scope="customer-a")
    store.write("customer B's secret", shard="episodic", scope="customer-b")

    result_a = store.search("query", scope="customer-a")
    result_b = store.search("query", scope="customer-b")

    assert "customer A's secret" in result_a.text
    assert "customer B's secret" not in result_a.text
    assert "customer B's secret" in result_b.text
    assert "customer A's secret" not in result_b.text


def test_shards_for_scope_never_returns_another_scopes_shard():
    store, embedder = _store()
    embedder.register("a", [1.0, 0.0, 0.0])
    embedder.register("b", [0.0, 1.0, 0.0])
    store.write("a", shard="episodic", scope="customer-a")
    store.write("b", shard="semantic", scope="customer-b")

    shards_a = store.shards_for_scope("customer-a")
    shards_b = store.shards_for_scope("customer-b")

    assert {s.id for s in shards_a} == {"episodic:customer-a"}
    assert {s.id for s in shards_b} == {"semantic:customer-b"}


def test_shards_for_scope_isolates_unscoped_from_scoped():
    store, embedder = _store()
    embedder.register("personal note", [1.0, 0.0, 0.0])
    embedder.register("tenant note", [0.0, 1.0, 0.0])
    store.write("personal note", shard="episodic", scope=None)
    store.write("tenant note", shard="episodic", scope="customer-a")

    unscoped_shards = store.shards_for_scope(None)
    scoped_shards = store.shards_for_scope("customer-a")

    assert {s.id for s in unscoped_shards} == {"episodic"}
    assert {s.id for s in scoped_shards} == {"episodic:customer-a"}


def test_search_respects_a_tight_budget():
    store, embedder = _store()
    embedder.register("memory", [1.0, 0.0, 0.0])
    embedder.register("query", [1.0, 0.0, 0.0])
    store.write("memory", shard="episodic")

    tiny_budget = Budget(max_tokens=0, max_latency_ms=0.0, max_usd=0.0)
    result = store.search("query", budget=tiny_budget)

    # query_all doesn't gate on budget mid-search (see lulu_router's own
    # strategy tests for that), but the trace should still carry the
    # budget through so /cost can report against it accurately.
    assert result.trace.budget == tiny_budget
