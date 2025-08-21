# services/ingest/connectors/worldbank.py
from __future__ import annotations
import json, requests
from typing import Dict, Any, List
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
        country: str = "all",        # e.g. "AT", "USA", "all"
        date: str = "2015:2024",     # e.g. "2018:2023" or "2023"
        per_page: int = 20000,       # large page to avoid pagination for most indicators
    ) -> Dict[str, Any]:
        """
        Returns {"meta": <dict>, "rows": <list>, "url": <str>}
        """
        url = (
            f"{self.BASE}/country/{country}/indicator/{indicator}"
            f"?format=json&per_page={per_page}&date={date}"
        )
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            raise ValueError(f"Unexpected WorldBank response shape: {type(data)}")
        meta, rows = data[0], data[1] or []
        return {"meta": meta, "rows": rows, "url": url}

    def normalize_row(self, row: Dict[str, Any], indicator: str) -> NormalizedDoc:
        country_name = (row.get("country") or {}).get("value") or ""
        iso3 = row.get("countryiso3code") or ""
        date = row.get("date") or ""
        value = row.get("value")

        value_str = "N/A" if value is None else str(value)
        title = f"{indicator} — {country_name or iso3 or 'Unknown'} — {date or 'n.d.'}"
        text = f"{indicator} for {country_name or iso3 or 'Unknown'} in {date or 'n.d.'}: {value_str}."

        link = f"{self.BASE}/country/{iso3 or 'all'}/indicator/{indicator}"

        core = make_core(
            source_id=source_id_kv(self.name, cca2 or name.lower()),
            source=self.name,
            doc_type="country_profile",
            title=f"{name} — profile",
            text=text,
            links=[...],
            country=[cca2, name] if cca2 or name else None,
            tags=["country", "profile"],
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
/*************  ✨ Windsurf Command ⭐  *************/
        """
        Normalize a WorldBank response dictionary into a list of NormalizedDoc objects.
        
        :param payload: A WorldBank API response dictionary, with a "rows" key
        :param indicator: The indicator ID (e.g., "NY.GDP.MKTP.CD")
        :return: A list of NormalizedDoc objects, one for each row in the response
        """
/*******  2a35aa21-0428-42e7-ab2a-2f4e84976e84  *******/
        rows = payload.get("rows") or []
        # Keep all rows (even None values) so you can decide downstream; or filter if you prefer:
        # rows = [r for r in rows if r.get("value") is not None]
        return [self.normalize_row(r, indicator) for r in rows]
