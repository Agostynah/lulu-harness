"""FakeEmbedder: maps registered strings to fixed, caller-chosen vectors,
so tests can construct scenarios with controlled similarity without
loading a real embedding model (no ONNX runtime, no model download, no
network -- consistent with every other fake in this test suite)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FakeEmbedder:
    vectors: dict[str, np.ndarray] = field(default_factory=dict)

    def register(self, text: str, vector) -> None:
        self.vectors[text] = np.asarray(vector, dtype=np.float32)

    def embed(self, text: str) -> np.ndarray:
        if text not in self.vectors:
            raise AssertionError(
                f"FakeEmbedder has no registered vector for {text!r} -- "
                "call .register(text, vector) in the test setup"
            )
        return self.vectors[text]


class ZeroEmbedder:
    """Returns the same fixed vector for any input, no registration
    needed. For tests that need a MemoryStore configured (so AgentLoop's
    memory path runs) but don't care about retrieval content -- e.g. CLI
    wiring tests, where FakeEmbedder's "raise on anything unregistered"
    strictness would just be test-setup noise."""

    def embed(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
