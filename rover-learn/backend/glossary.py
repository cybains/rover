import csv
from pathlib import Path
from typing import List, Dict


def load_glossary() -> List[Dict[str, str]]:
    path = Path(__file__).resolve().parent.parent / "config" / "glossary.csv"
    items: List[Dict[str, str]] = []
    if not path.exists():
        return items
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            items.append({
                "term": row.get("term", ""),
                "de": row.get("de", ""),
                "en": row.get("en", ""),
                "notes": row.get("notes", ""),
            })
    return items
