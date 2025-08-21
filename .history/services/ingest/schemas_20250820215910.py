from __future__ import annotations
from dataclasses import dataclass, asdict   # <-- this is missing
from typing import Any, Dict, List, Optional
from datetime import datetime


# ---- Core (universal) ------------------------------------------------------


@dataclass
class CoreDoc:
    source_id: str # stable ID for upserts (e.g., "url:https://...", "pdf:client/file.pdf")
source: str # producer name (e.g., "restcountries", "crawler", "pdf_upload")
doc_type: str # e.g., "article", "fx", "pdf", "country_profile"
title: str # short human title
text: str # normalized human-readable body (what we chunk/embed)
links: List[str] # URLs or file refs for provenance/citation
timestamp: str # ISO-8601 when (re)ingested
version: int = 1


# common filter metadata (optional but recommended)
country: Optional[List[str]] = None
tags: Optional[List[str]] = None


# change detection + audit
content_hash: Optional[str] = None
valid_for: Optional[Dict[str, Optional[str]]] = None # {"from": str | None, "to": str | None}


# the facet + raw are attached outside of CoreDoc in NormalizedDoc


@dataclass
class NormalizedDoc:
    core: CoreDoc
facet: Dict[str, Any] # doc_type-specific payload (see facets.py)
raw: Optional[Dict[str, Any]] = None # trimmed "as-fetched" blob for audit/debug


def to_dict(self) -> Dict[str, Any]:
    d = {
**asdict(self.core),
"facet": self.facet,
"raw": self.raw or {},
}
    return d




def now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"




def make_core(
*,
source_id: str,
source: str,
doc_type: str,
title: str,
text: str,
links: List[str],
country: Optional[List[str]] = None,
tags: Optional[List[str]] = None,
version: int = 1,
content_hash: Optional[str] = None,
valid_from: Optional[str] = None,
valid_to: Optional[str] = None,
) -> CoreDoc:
    return CoreDoc(
source_id=source_id,
source=source,
doc_type=doc_type,
title=title,
text=text,
links=links,
timestamp=now_iso(),
version=version,
country=country,
tags=tags,
content_hash=content_hash,
valid_for={"from": valid_from, "to": valid_to} if (valid_from or valid_to) else None,
)