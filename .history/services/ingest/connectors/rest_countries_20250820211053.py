from __future__ import annotations
import json
import requests
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


core = make_core(
source_id=source_id_kv(self.name, cca2 or name.lower()),
source=self.name,
doc_type="country_profile",
title=f"{name} — profile",
text=text,
links=[f"https://restcountries.com/v3.1/alpha/{cca2}" if cca2 else self.URL],
country=[cca2, name] if cca2 or name else None,
tags=["country", "profile"],
)


# For an API doc, we could use an 'article' facet minimal placeholder.
facet = facet_article(url=self.URL, canonical=None, headings=[name], author=None, published_at=None)


# Compute content hash (text + facet JSON)
facet_str = json.dumps(facet, ensure_ascii=False, sort_keys=True)
core.content_hash = content_hash(core.text, facet_str)return NormalizedDoc(core=core, facet=facet, raw=r)