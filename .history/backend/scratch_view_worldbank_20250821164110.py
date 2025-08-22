# scratch_view_worldbank.py
import json
from dataclasses import is_dataclass, asdict

from services.ingest.connectors.worldbank import WorldBank

def to_dict(x):
    """Best-effort serializer for core/facet/raw regardless of schema impl."""
    # dataclass?
    if is_dataclass(x):
        return asdict(x)
    # pydantic v1/v2?
    for m in ("model_dump", "dict"):
        if hasattr(x, m):
            try:
                return getattr(x, m)()
            except TypeError:
                return getattr(x, m)(exclude_none=True)
    # plain object with __dict__
    if hasattr(x, "__dict__"):
        return x.__dict__
    # already a mapping or list/primitive
    return x

def main():
    wb = WorldBank()
    indicator = "NY.GDP.MKTP.CD"   # GDP (current US$)
    payload = wb.fetch_indicator(indicator=indicator, country="AT", date="2022:2024")
    docs = wb.normalize(payload, indicator=indicator)

    # Show a compact preview
    for d in docs[:5]:
        print("—", d.core.title)
        print("  ", d.core.text)
        print()

    # Optionally dump the first 5 docs as JSON to inspect full schema
    out = []
    for d in docs[:5]:
        out.append({
            "core":  to_dict(d.core),
            "facet": to_dict(d.facet),
            "raw":   to_dict(d.raw),
        })
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
