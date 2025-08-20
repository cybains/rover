import hashlib
from typing import Optional




def content_hash(text: str, facet_json: Optional[str] = None) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    if facet_json: 
        h.update(b"\x00")
        h.update(facet_json.encode("utf-8"))
return h.hexdigest()