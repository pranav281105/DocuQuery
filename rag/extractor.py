"""
rag/extractor.py — Extract text from a PDF, page by page.

Returns a list of dicts: [{"page": int, "text": str}, ...]
Page numbers are 1-indexed to match what users see in their PDF viewer.
"""

import fitz  # PyMuPDF


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    """
    Extract text from a PDF supplied as raw bytes.

    Args:
        pdf_bytes: Raw bytes of the PDF file.

    Returns:
        List of page dicts, e.g. [{"page": 1, "text": "..."}, ...]

    Raises:
        ValueError: If the PDF has no extractable text.
        RuntimeError: If PyMuPDF cannot open the file.
    """
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as exc:
        raise RuntimeError(f"Could not open PDF: {exc}") from exc

    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i, "text": text})

    doc.close()

    if not pages:
        raise ValueError(
            "No extractable text found. The PDF may be scanned or image-only."
        )

    return pages
