# services/ingest/schemas.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, List, Optional


@dataclass
class CoreDoc:
    """
    Minimal core document schema for your normalized records.
    """
    source_id: str
    doc_type: str
    title: str
    text: str
    links: List[str]
    country: Optional[list] = None
    tags: Optional[list] = None
    content_hash: Optional[str] = None  # to be filled post-creation if needed


@dataclass
class NormalizedDoc:
    """
    Wrapper the connectors return: core fields + facet/provenance + raw source record.
    """
    core: CoreDoc
    facet: Any   # whatever facet_article(...) returns (usually a dict)
    raw: Any     # original row / payload from the source


def make_core(
    *,
    source_id: str,
    doc_type: str,
    title: str,
    text: str,
    links: List[str],
    country: Optional[list] = None,
    tags: Optional[list] = None,
    source: Optional[str] = None,  # accepted for backward/forward compat, not stored directly
) -> CoreDoc:
    """
    Factory to create CoreDoc while gracefully handling a 'source' kwarg.
    We don't persist 'source' as a CoreDoc field; if provided, we fold it into tags.
    """
    # Merge 'source' into tags for provenance without altering CoreDoc signature
    if source:
        tags = list(tags) if tags else []
        if source not in tags:
            tags = [source, *tags]

    return CoreDoc(
        source_id=source_id,
        doc_type=doc_type,
        title=title,
        text=text,
        links=links,
        country=country,
        tags=tags,
        # content_hash can be added later by the connector (e.g., after computing from text+facet)
    )


__all__ = ["CoreDoc", "NormalizedDoc", "make_core"]
