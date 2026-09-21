"""SQLiteShardStore: the persistent ShardStore backend. Every test uses
tmp_path so nothing touches a real file outside pytest's own sandbox."""

from __future__ import annotations

import numpy as np
import pytest

from lulu_router.backends.sqlite import SQLiteShardStore


def test_empty_store_has_zero_length(tmp_path):
    store = SQLiteShardStore(tmp_path / "episodic.db")
    assert len(store) == 0


def test_add_then_search_finds_the_closest_match(tmp_path):
    store = SQLiteShardStore(tmp_path / "episodic.db")
    store.add("a", "decided to use SQLite for the cache", np.array([1.0, 0.0, 0.0]))
    store.add("b", "the weather today is sunny", np.array([0.0, 1.0, 0.0]))

    results = store.search(np.array([1.0, 0.0, 0.0]), k=1)

    assert len(results) == 1
    assert results[0].id == "a"
    assert results[0].content == "decided to use SQLite for the cache"


def test_add_normalizes_the_stored_vector(tmp_path):
    store = SQLiteShardStore(tmp_path / "episodic.db")
    store.add("a", "x", np.array([3.0, 4.0, 0.0]))  # norm = 5
    results = store.search(np.array([1.0, 0.0, 0.0]), k=1)
    assert results[0].score == pytest.approx(3.0 / 5.0, abs=1e-5)


def test_len_reflects_row_count(tmp_path):
    store = SQLiteShardStore(tmp_path / "episodic.db")
    store.add("a", "x", np.array([1.0, 0.0]))
    store.add("b", "y", np.array([0.0, 1.0]))
    assert len(store) == 2


def test_data_survives_reopening_the_same_db_file(tmp_path):
    db_path = tmp_path / "episodic.db"
    store = SQLiteShardStore(db_path)
    store.add("a", "persisted memory", np.array([1.0, 0.0, 0.0]))
    del store  # simulate the process exiting

    reopened = SQLiteShardStore(db_path)
    assert len(reopened) == 1
    results = reopened.search(np.array([1.0, 0.0, 0.0]), k=1)
    assert results[0].content == "persisted memory"


def test_centroid_is_none_when_empty(tmp_path):
    store = SQLiteShardStore(tmp_path / "episodic.db")
    assert store.centroid() is None


def test_centroid_is_mean_of_stored_vectors(tmp_path):
    store = SQLiteShardStore(tmp_path / "episodic.db")
    store.add("a", "x", np.array([1.0, 0.0]))
    store.add("b", "y", np.array([0.0, 1.0]))
    centroid = store.centroid()
    assert centroid is not None
    assert float(centroid[0]) == pytest.approx(0.5, abs=1e-5)
    assert float(centroid[1]) == pytest.approx(0.5, abs=1e-5)


def test_search_on_a_fresh_read_reflects_writes_from_a_different_store_instance(tmp_path):
    # Exercises the read cache's invalidation across instances, not just
    # within one -- two SQLiteShardStore objects pointed at the same file
    # (e.g. two MemoryStore instances in the same process, or a restart)
    # must never see stale data from before either of them last wrote.
    db_path = tmp_path / "episodic.db"
    writer = SQLiteShardStore(db_path)
    writer.add("a", "first", np.array([1.0, 0.0]))

    reader = SQLiteShardStore(db_path)
    assert len(reader.search(np.array([1.0, 0.0]), k=5)) == 1
