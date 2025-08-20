from __future__ import annotations
from typing import Dict, Any
from bs4 import BeautifulSoup # pip install beautifulsoup4
import requests
from ..schemas import make_core, NormalizedDoc
from ..facets import facet_article
from ..utils_id import source_id_url
from ..utils_hash import content_hash
import json




class SimpleCrawler:
name = "crawler"


def fetch(self, url: str):
r = requests.get(url, timeout=30)
r.raise_for_status()
return [{"url": url, "html": r.text}]


def _extract(self, html: str) -> Dict[str, Any]:
soup = BeautifulSoup(html, "html.parser")
title = (soup.title.string if soup.title else "").strip()
# naive text extraction; replace with readability later
text = "\n".join([t.get_text(" ", strip=True) for t in soup.find_all(["h1","h2","h3","p","li"])])
headings = [h.get_text(" ", strip=True) for h in soup.find_all(["h1","h2","h3"])]
return {"title": title, "text": text, "headings": headings}


def normalize(self, payload: Dict[str, Any]) -> NormalizedDoc:
url = payload["url"]
ext = self._extract(payload["html"])
core = make_core(
source_id=source_id_url(url),
source=self.name,
doc_type="article",
title=ext["title"] or url,
text=ext["text"],
links=[url],
tags=["web"],
)
facet = facet_article(url=url, canonical=url, headings=ext["headings"], author=None, published_at=None)
core.content_hash = content_hash(core.text, json.dumps(facet, ensure_ascii=False, sort_keys=True))
return NormalizedDoc(core=core, facet=facet, raw={"url": url})