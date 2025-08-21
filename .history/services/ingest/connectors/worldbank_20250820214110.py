from __future__ import annotations
import json, time, requests
from typing import Dict, Any, Iterable, List, Optional
from ..schemas import make_core, NormalizedDoc
from ..facets import facet_worldbank
from ..utils_id import source_id_kv
from ..utils_hash import content_hash

BASE_URL = "https://search.worldbank.org/api/v3/wds"
ROWS_PER_PAGE = 50  # API max

def _iter_pages(qterm: Optional[str], max_pages: Optional[int]) -> Iterable[Dict[str, Any]]:
    offset = 0
    page = 0
    while True:
        params = {"format": "json", "rows": ROWS_PER_PAGE, "os": offset}
        if qterm:
            params["qterm"] = qterm
        r = requests.get(BASE_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        docs = data.get("documents", {}) or {}
        if not docs:
            break
        yield from docs.values()

        page += 1
        offset += ROWS_PER_PAGE
        total = int(data.get("total", 0))
        if max_pages is not None and page >= max_pages:
            break
        if offset >= total:
            break
        time.sleep(0.8)  # be nice

class WorldBank:
    name = "worldbank"

    def fetch(self, qterm: Optional[str] = None, max_pages: Optional[int] = None) -> Iterable[Dict[str, Any]]:
        return list(_iter_pages(qterm, max_pages))

    def normalize(self, r: Dict[str, Any]) -> NormalizedDoc:
        # Common WB fields (defensive defaults)
        doc_id = str(r.get("id") or r.get("docna") or r.get("url") or "")
        title = (r.get("title") or r.get("docdt") or "World Bank Document").strip()
        abstract = (r.get("abstracts") or r.get("abstract") or "").strip()
        url = (r.get("url") or r.get("urllinktext") or "").strip()
        pubyear = (r.get("displayconttype") or r.get("docdt") or r.get("theme") or None)
        countries = r.get("countryshortname") or r.get("countryname") or r.get("country") or []
        if isinstance(countries, str):
            countries = [countries]
        topics = r.get("majorsector") or r.get("topic") or []
        if isinstance(topics, str):
            topics = [topics]

        # Build human-readable text for retrieval (title + abstract as primary signal)
        body = f"{title}\n\n{abstract}".strip() or title

        core = make_core(
            source_id=source_id_kv(self.name, doc_id),
            source=self.name,
            doc_type="wb_doc",
            title=title,
            text=body,
            links=[url] if url else [],
            country=countries or None,
            tags=["worldbank", "report"] + ([t for t in topics if isinstance(t, str)] or []),
        )

        facet = facet_worldbank(
            doc_id=doc_id, url=url, pubyear=str(pubyear) if pubyear else None,
            countries=[c for c in countries if isinstance(c, str)],
            topics=[t for t in topics if isinstance(t, str)],
        )

        facet_str = json.dumps(facet, ensure_ascii=False, sort_keys=True)
        core.content_hash = content_hash(core.text, facet_str)

        return NormalizedDoc(core=core, facet=facet, raw=r)
