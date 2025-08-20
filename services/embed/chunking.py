from __future__ import annotations
from typing import Dict, Iterable, List


def chunk(core_doc: Dict, max_chars: int = 1000, overlap: int = 120) -> List[Dict]:
    """Split a document into overlapping text chunks.

    The previous implementation built an intermediate list of string parts before
    constructing chunk dictionaries.  This version constructs the chunk metadata
    directly during iteration to reduce memory usage and simplify the logic.
    """

    text = core_doc["text"]
    if not text:
        return []

    n = len(text)
    chunks: List[Dict] = []
    start = 0
    i = 0
    while True:
        end = min(start + max_chars, n)
        part = text[start:end]

        chunks.append(
            {
                "doc_id": core_doc["source_id"],
                "chunk_id": f"{core_doc['source_id']}::{i}",
                "text": part,
                "metadata": {
                    k: core_doc.get(k)
                    for k in ("source", "doc_type", "country", "tags", "title")
                },
            }
        )

        if end == n:
            break
        start = max(end - overlap, 0)
        i += 1

    return chunks


def iter_chunks(normalized_docs: Iterable[Dict]) -> Iterable[Dict]:
    """Yield chunks for a stream of normalized documents."""

    for nd in normalized_docs:
        core = {k: v for k, v in nd.items() if k not in ("facet", "raw")}
        for c in chunk(core):
            yield c
