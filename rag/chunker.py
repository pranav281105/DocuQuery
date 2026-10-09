"""
rag/chunker.py — Split page text into overlapping chunks while preserving page metadata.

Chunks are created with a sliding window over words to ensure context continuity
at chunk boundaries.
"""


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 400,
    overlap: int = 80,
) -> list[dict]:
    """
    Split page text into overlapping word-based chunks.

    Args:
        pages:      Output of extractor.extract_pages().
        chunk_size: Target number of words per chunk.
        overlap:    Number of words shared between consecutive chunks.

    Returns:
        List of chunk dicts: [{"page": int, "text": str, "chunk_index": int}, ...]
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    chunks = []
    chunk_index = 0

    for page_dict in pages:
        words = page_dict["text"].split()
        if not words:
            continue

        step = chunk_size - overlap
        start = 0
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(
                {
                    "page": page_dict["page"],
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                }
            )
            chunk_index += 1
            if end == len(words):
                break
            start += step

    return chunks
