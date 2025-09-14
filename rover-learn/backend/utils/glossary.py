from __future__ import annotations

import csv
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple


@lru_cache()
def load_glossary() -> List[dict]:
    path = Path(__file__).resolve().parents[1] / "config" / "glossary.csv"
    items: List[dict] = []
    if not path.exists():
        return items
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            term_id = row.get("term") or row.get("id") or ""
            en = (row.get("en") or "").strip()
            items.append({"id": term_id, "en": en})
    return items


def apply_glossary(text: str) -> Tuple[str, List[str]]:
    items = load_glossary()
    hits: List[str] = []
    out = text
    for item in items:
        term = item.get("en")
        tid = item.get("id")
        if not term or not tid:
            continue
        pat = re.compile(rf"\b{re.escape(term)}\b", re.I)
        if pat.search(out):
            hits.append(tid)
            out = pat.sub(term, out)
    return out, hits
