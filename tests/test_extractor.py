"""
tests/test_extractor.py — Unit tests for the PDF extractor.

Uses PyMuPDF to create minimal in-memory PDFs so tests run without fixtures on disk.
"""

import pytest
import fitz  # PyMuPDF

from rag.extractor import extract_pages


def _make_pdf(page_texts: list[str]) -> bytes:
    """Create a minimal PDF in memory with the given text on each page."""
    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text, fontsize=12)
    buf = doc.tobytes()
    doc.close()
    return buf


def test_single_page_extraction():
    pdf = _make_pdf(["Hello from page one."])
    pages = extract_pages(pdf)
    assert len(pages) == 1
    assert pages[0]["page"] == 1
    assert "Hello" in pages[0]["text"]


def test_multi_page_extraction():
    pdf = _make_pdf(["Page one text.", "Page two text.", "Page three text."])
    pages = extract_pages(pdf)
    assert len(pages) == 3
    assert pages[0]["page"] == 1
    assert pages[2]["page"] == 3


def test_page_numbers_are_one_indexed():
    pdf = _make_pdf(["First", "Second"])
    pages = extract_pages(pdf)
    assert pages[0]["page"] == 1
    assert pages[1]["page"] == 2


def test_empty_pdf_raises_value_error():
    pdf = _make_pdf([""])  # blank page — no extractable text
    with pytest.raises(ValueError, match="No extractable text"):
        extract_pages(pdf)


def test_invalid_bytes_raises_runtime_error():
    with pytest.raises(RuntimeError, match="Could not open PDF"):
        extract_pages(b"not a pdf")
