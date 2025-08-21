from __future__ import annotations
import json, requests
from typing import Dict, Any
from ..schemas import make_core, NormalizedDoc
from ..facets import facet_article
from ..utils_id import source_id_kv
from ..utils_hash import content_hash

class RestCountries:
    name = "restcountries"
    URL = "https://restcountries.com/v3.1/all"

    def fetch(self):
        r = requests.get(self.URL, timeout=30)
        r.raise_for_status()
        return r.json()

    def normalize(self, r: Dict[str, Any]) -> NormalizedDoc:
        name = (r.get("name") or {}).get("common", "")
        cca2 = r.get("cca2", "")
        region = r.get("region", "")
        capital = ", ".join(r.get("capital", []) or [])
        langs = ", ".join((r.get("languages") or {}).values() or [])
        currs = ", ".join((r.get("currencies") or {}).keys() or [])

        text = (
            f"{name} ({cca2}) is in {region}. "
            f"Capital: {capital or 'N/A'}. "
            f"Languages: {langs or 'N/A'}. Currencies: {currs or 'N/A'}."
        )

        
        

        facet = facet_article(
            url=self.URL, canonical=None, headings=[name],
            author=None, published_at=None
        )

        facet_str = json.dumps(facet, ensure_ascii=False, sort_keys=True)
        core.content_hash = content_hash(core.text, facet_str)

        # ✅ This return must be indented inside normalize()
        return NormalizedDoc(core=core, facet=facet, raw=r)
