"""
tests/test_embedder.py — Unit tests for embedding and retrieval.

Uses a real SentenceTransformer model. Tests are intentionally lightweight
(small sentences) to keep CI fast.
"""

import numpy as np
import pytest

from rag.embedder import embed_chunks, load_model, retrieve


@pytest.fixture(scope="module")
def model():
    return load_model()


def _chunks(texts: list[str], page: int = 1) -> list[dict]:
    return [
        {"page": page, "text": t, "chunk_index": i} for i, t in enumerate(texts)
    ]


def test_embed_returns_correct_shape(model):
    texts = ["hello world", "foo bar baz"]
    chunks = _chunks(texts)
    embs = embed_chunks(model, chunks)
    assert embs.shape[0] == 2
    assert embs.ndim == 2


def test_embed_returns_float32(model):
    chunks = _chunks(["test sentence"])
    embs = embed_chunks(model, chunks)
    assert embs.dtype == np.float32


def test_retrieve_returns_top_k(model):
    chunks = _chunks(["cats and dogs", "quantum physics", "machine learning", "weather forecast"])
    embs = embed_chunks(model, chunks)
    results = retrieve(model, "animals", chunks, embs, top_k=2)
    assert len(results) == 2


def test_retrieve_adds_score(model):
    chunks = _chunks(["solar system planets", "cooking recipes"])
    embs = embed_chunks(model, chunks)
    results = retrieve(model, "astronomy", chunks, embs, top_k=2)
    for r in results:
        assert "score" in r
        assert -1.0 <= r["score"] <= 1.0


def test_retrieve_relevant_chunk_ranks_higher(model):
    chunks = _chunks(["The capital of France is Paris.", "Water boils at 100 degrees."])
    embs = embed_chunks(model, chunks)
    results = retrieve(model, "What is the capital of France?", chunks, embs, top_k=2)
    assert "Paris" in results[0]["text"]


def test_retrieve_top_k_capped_at_chunk_count(model):
    chunks = _chunks(["only one chunk"])
    embs = embed_chunks(model, chunks)
    results = retrieve(model, "query", chunks, embs, top_k=10)
    assert len(results) == 1
