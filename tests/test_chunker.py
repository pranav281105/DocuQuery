"""
tests/test_chunker.py — Unit tests for the chunking module.
"""

import pytest

from rag.chunker import chunk_pages


def _make_pages(texts: list[str]) -> list[dict]:
    return [{"page": i + 1, "text": t} for i, t in enumerate(texts)]


def test_single_page_single_chunk():
    pages = _make_pages(["word " * 10])
    chunks = chunk_pages(pages, chunk_size=20, overlap=5)
    assert len(chunks) == 1
    assert chunks[0]["page"] == 1
    assert chunks[0]["chunk_index"] == 0


def test_overlap_creates_multiple_chunks():
    # 30 words, chunk_size=20, overlap=5 → step=15 → chunks at 0 and 15 → 2 chunks
    pages = _make_pages(["word " * 30])
    chunks = chunk_pages(pages, chunk_size=20, overlap=5)
    assert len(chunks) == 2


def test_page_number_preserved():
    pages = _make_pages(["alpha " * 5, "beta " * 5])
    chunks = chunk_pages(pages, chunk_size=10, overlap=0)
    page_numbers = {c["page"] for c in chunks}
    assert 1 in page_numbers
    assert 2 in page_numbers


def test_chunk_index_monotonic():
    pages = _make_pages(["word " * 50])
    chunks = chunk_pages(pages, chunk_size=10, overlap=2)
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(chunks)))


def test_empty_pages_skipped():
    pages = _make_pages(["", "   ", "content here"])
    chunks = chunk_pages(pages, chunk_size=10, overlap=2)
    # Only the non-empty page should produce chunks
    assert all(c["page"] == 3 for c in chunks)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError, match="chunk_size must be greater than overlap"):
        chunk_pages(_make_pages(["text"]), chunk_size=5, overlap=5)


def test_chunk_text_contains_words():
    pages = _make_pages(["hello world foo bar baz"])
    chunks = chunk_pages(pages, chunk_size=3, overlap=1)
    for chunk in chunks:
        assert chunk["text"].strip() != ""
