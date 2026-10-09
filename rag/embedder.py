"""
rag/embedder.py — Generate embeddings and retrieve top-k chunks by cosine similarity.

Uses a lightweight Sentence Transformers model that runs locally with no API key.
"""

import numpy as np
from sentence_transformers import SentenceTransformer

# Loaded once per process; cached by Streamlit's @st.cache_resource in app.py.
_MODEL_NAME = "all-MiniLM-L6-v2"


def load_model() -> SentenceTransformer:
    """Return the embedding model (call once and cache the result)."""
    return SentenceTransformer(_MODEL_NAME)


def embed_chunks(model: SentenceTransformer, chunks: list[dict]) -> np.ndarray:
    """
    Embed a list of chunk dicts.

    Args:
        model:  SentenceTransformer instance.
        chunks: List of chunk dicts from chunker.chunk_pages().

    Returns:
        2-D float32 array of shape (n_chunks, embedding_dim).
    """
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(np.float32)


def retrieve(
    model: SentenceTransformer,
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    top_k: int = 5,
) -> list[dict]:
    """
    Retrieve the top_k most relevant chunks for a query using cosine similarity.

    Args:
        model:      SentenceTransformer instance.
        query:      User's question string.
        chunks:     Original chunk dicts (parallel to embeddings).
        embeddings: Pre-computed chunk embeddings.
        top_k:      Number of chunks to return.

    Returns:
        List of chunk dicts sorted by descending relevance, each augmented with
        a "score" key (float in [-1, 1]).
    """
    query_vec = model.encode([query], convert_to_numpy=True).astype(np.float32)

    # Cosine similarity: dot product of L2-normalised vectors.
    chunk_norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_norm = np.linalg.norm(query_vec)

    # Guard against zero vectors.
    safe_chunk_norms = np.where(chunk_norms == 0, 1.0, chunk_norms)
    safe_query_norm = query_norm if query_norm != 0 else 1.0

    normalised_chunks = embeddings / safe_chunk_norms
    normalised_query = query_vec / safe_query_norm

    scores = (normalised_chunks @ normalised_query.T).flatten()

    k = min(top_k, len(chunks))
    top_indices = np.argsort(scores)[::-1][:k]

    results = []
    for idx in top_indices:
        result = dict(chunks[idx])
        result["score"] = float(scores[idx])
        results.append(result)

    return results
