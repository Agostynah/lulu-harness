"""embeddings.py: turns text into vectors for the memory router.

Lives in lulu-core, not lulu-router, deliberately: the router is
embedding-agnostic by design (every ShardStore.search() takes an
already-computed query_vec) so it can be tested and benchmarked
(evals/dbpedia) with synthetic vectors and no ONNX runtime required. The
harness is what actually has to turn real text into vectors, so this
wrapper -- and the model download it implies -- lives here instead.

Lazy-loaded: constructing an Embedder does not load the model or touch
the network; the first call to embed() does. Tests inject a fake instead
of touching this at all (see MemoryStore's `embedder` param).
"""

from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DIM = 384


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None

    def _load(self) -> Any:
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(self.model_name)
        return self._model

    def embed(self, text: str) -> np.ndarray:
        model = self._load()
        vec = next(iter(model.embed([text])))
        return np.asarray(vec, dtype=np.float32)
