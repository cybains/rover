from __future__ import annotations
from typing import Dict, Any, List, Optional


# Facet shapes by doc_type. Keep them lightweight and evolve as needed.


# fx facet (e.g., ECB/hosted rates)
def facet_fx(*, date: str, base_currency: str, rates: Dict[str, float]) -> Dict[str, Any]:
return {"fx": {"date": date, "base_currency": base_currency, "rates": rates}}


# article/web facet


def facet_article(*, url: str, canonical: Optional[str], headings: List[str], author: Optional[str], published_at: Optional[str]) -> Dict[str, Any]:
return {
"article": {
"url": url,
"canonical": canonical,
"headings": headings,
"author": author,
"published_at": published_at,
}
}


# pdf facet


def facet_pdf(*, filename: str, client_id: Optional[str], pages: int, mime: str) -> Dict[str, Any]:
return {"pdf": {"filename": filename, "client_id": client_id, "pages": pages, "mime": mime}}


# indicator facet (eurostat-style)


def facet_indicator(*, dataset_id: str, unit: str, geo: str, period: str, value: float) -> Dict[str, Any]:
return {"indicator": {"dataset_id": dataset_id, "unit": unit, "geo": geo, "period": period, "value": value}}