import hashlib
from typing import Optional


def content_hash(text: str, facet_json: Optional[str] = None) -> str:
    """Compute a stable hash for a document's text and optional facet."""

    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    if facet_json:
        # Separate the facet content from the text with a null byte so that
        # `text='ab', facet='c'` does not collide with `text='a', facet='bc'`.
        h.update(b"\x00")
        h.update(facet_json.encode("utf-8"))

    return h.hexdigest()
