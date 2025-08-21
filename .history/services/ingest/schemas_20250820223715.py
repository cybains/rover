# services/ingest/schemas.py
from typing import Optional, List

def make_core(
    *,
    source_id: str,
    doc_type: str,
    title: str,
    text: str,
    links: List[str],
    country: Optional[list] = None,
    tags: Optional[list] = None,
    source: Optional[str] = None,  # keep but don't forward to CoreDoc
):
    # If you want to keep provenance, merge source into tags safely.
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
        # IMPORTANT: do NOT pass 'source' to CoreDoc here
    )
