# services/ingest/connectors/worldbank.py
from __future__ import annotations
import json
from typing import Any, Dict, List, Optional
import requests

from ..schemas import make_core, NormalizedDoc
from ..facets import facet_article
from ..utils_id import source_id_kv
from ..utils_hash import content_hash


class WorldBank:
    name = "worldbank"
    BASE = "https://api.worldbank.org/v2"

    def fetch_indicator(
        self,
        indicator: str,
        country: str = "all",       # e.g. "AT", "USA", "all"
        date: str = "2018:2024",    # e.g. "2023" or "2018:2024"
        per_page: int = 1000,
        all_pages: bool = True,
        timeout: int = 60,
    ) -> Dict[str, Any]:
        """
        Fetch World Bank indicator data.

        Returns:
            {
              "meta": <dict>,
              "rows": <list of row dicts>,
              "url":  <first page URL for traceability>
            }
        """
        page = 1
        url = (
            f"{self.BASE}/country/{country}/indicator/{indicator}"
            f"?format=json&date={date}&per_page={per_page}&page={page}"
        )
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            raise ValueError(f"Unexpected WorldBank response: {type(data)} {data!r}")

        meta, rows = data[0], data[1] or []

        # Handle pagination if requested
        total_pages = int(meta.get("pages") or 1)
        if all_pages and total_pages > 1:
            for page in range(2, total_pages + 1):
                page_url = (
                    f"{self.BASE}/country/{country}/indicator/{indicator}"
                    f"?format=json&date={date}&per_page={per_page}&page={page}"
                )
                rp = requests.get(page_url, timeout=timeout)
                rp.raise_for_status()
                dp = rp.json()
                if isinstance(dp, list) and len(dp) >= 2 and isinstance(dp[1], list):
                    rows.extend(dp[1])

        return {"meta": meta, "rows": rows, "url": url}

    def normalize_row(self, row: Dict[str, Any], indicator: str) -> NormalizedDoc:
        """
        Convert a single World Bank row into your NormalizedDoc.
        Typical row keys include:
          - country: {"id": "...", "value": "Austria"}
          - countryiso3code: "AUT"
          - date: "2023"
          - value: number or None
        """
        country_name = (row.get("country") or {}).get("value") or ""
        iso3 = row.get("countryiso3code") or ""
        date = row.get("date") or ""
        value = row.get("value")

        value_str = "N/A" if value is None else str(value)
        title = f"{indicator} — {country_name or iso3 or 'Unknown'} — {date or 'n.d.'}"
        text = f"{indicator} for {country_name or iso3 or 'Unknown'} in {date or 'n.d.'}: {value_str}."

        link = f"{self.BASE}/country/{iso3 or 'all'}/indicator/{indicator}"

        core = make_core(
            source_id=source_id_kv(self.name, f"{indicator}:{iso3 or 'ALL'}:{date or 'NA'}"),
            doc_type="indicator_observation",
            title=title,
            text=text,
            links=[link],
            country=[iso3, country_name] if iso3 or country_name else None,
            tags=["worldbank", "indicator", indicator],
        )

        facet = facet_article(
            url=link,
            canonical=None,
            headings=[indicator, country_name or iso3 or "Unknown"],
            author=None,
            published_at=None,
        )

        facet_str = json.dumps(facet, ensure_ascii=False, sort_keys=True)
        core.content_hash = content_hash(core.text, facet_str)

        return NormalizedDoc(core=core, facet=facet, raw=row)

    def normalize(self, payload: Dict[str, Any], indicator: str) -> List[NormalizedDoc]:
        """Normalize an entire payload returned by fetch_indicator()."""
        rows = payload.get("rows") or []
        return [self.normalize_row(r, indicator) for r in rows]


__all__ = ["WorldBank"]
