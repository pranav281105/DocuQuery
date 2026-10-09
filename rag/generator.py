"""
rag/generator.py — Generate an answer using the Gemini API grounded in retrieved context.
"""

import os

from google import genai
from google.genai import types


def _get_api_key() -> str:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before starting the application."
        )
    return key


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """
    Ask Gemini to answer a question using only the supplied context chunks.

    Args:
        query:          The user's question.
        context_chunks: Retrieved chunk dicts (each has "page" and "text" keys).

    Returns:
        The model's answer as a plain string.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is not set.
        RuntimeError:     If the Gemini API call fails.
    """
    client = genai.Client(api_key=_get_api_key())

    # Build context block with page references so Gemini can cite them naturally.
    context_parts = []
    for chunk in context_chunks:
        context_parts.append(f"[Page {chunk['page']}]\n{chunk['text']}")
    context_block = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are a document assistant. Answer the question using ONLY the "
        "document excerpts provided below. If the answer is not present in "
        "the excerpts, say: \"I could not find an answer in the provided document.\"\n\n"
        f"=== Document Excerpts ===\n{context_block}\n\n"
        f"=== Question ===\n{query}\n\n"
        "=== Answer ==="
    )

    try:
        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
        )
        return response.text.strip()
    except Exception as exc:
        raise RuntimeError(f"Gemini API error: {exc}") from exc
