# services/ingest/run_ingest.py — updated
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List

# If you always run via `python -m services.ingest.run_ingest ...`
# you don't need to modify sys.path. Keep it only if you invoke directly.
# import sys
# sys.path.append(str(Path(__file__).resolve().parents[2]))

from services.ingest.connectors.rest_countries import RestCountries
from services.ingest.connectors.worldbank import WorldBank
# Optional stubs (leave commented if not in use right now)
# from services.ingest.connectors.web_crawler_stub import SimpleCrawler
# from services.ingest.connectors.pdf_intake_stub import PdfIntake


# -----------------
# Paths & constants
# -----------------
DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DOCS_PATH = DATA_ROOT / "docs.jsonl"
STATE_PATH = DATA_ROOT / "state.last_hash.json"
DATA_ROOT.mkdir(parents=True, exist_ok=True)


# -----------------
# State I/O helpers
# -----------------
def load_state() -> Dict[str, str]:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: Dict[str, str]) -> None:
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def obj_to_dict(obj):
    """Serialize dataclasses or plain objects to a JSON-friendly dict."""
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return obj


def append_docs(docs: Iterable) -> None:
    with DOCS_PATH.open("a", encoding="utf-8") as f:
        for nd in docs:
            nd_dict = obj_to_dict(nd)
            f.write(json.dumps(nd_dict, ensure_ascii=False) + "\n")


# -------------
# Main routine
# -------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Run rover ingestion")
    ap.add_argument(
        "connector",
        choices=["rest_countries", "worldbank"],  # add "crawl", "pdf" later when needed
        help="Which connector to run",
    )

    # WorldBank-specific args
    ap.add_argument("--indicator", help="World Bank indicator id (e.g., NY.GDP.MKTP.CD)")
    ap.add_argument("--country", help="ISO2 or 'all' (e.g., AT, USA, all)")
    ap.add_argument("--date", help="Year or range (e.g., 2022 or 2018:2023)")
    ap.add_argument("--per-page", type=int, default=1000, help="WB per_page size")

    # Uncomment if you (later) re-enable crawler/pdf stubs
    # ap.add_argument("--url", help="for crawl connector")
    # ap.add_argument("--pdf", help="for pdf connector (file path)")

    args = ap.parse_args()

    state = load_state()
    changed: List = []

    if args.connector == "rest_countries":
        rc = RestCountries()
        payloads = rc.fetch()  # list[dict]
        # Normalize each payload into a NormalizedDoc
        docs = [rc.normalize(p) for p in payloads]

    elif args.connector == "worldbank":
        # Sensible defaults if flags omitted
        indicator = args.indicator or "NY.GDP.MKTP.CD"
        country = args.country or "AT"
        date = args.date or "2018:2023"

        wb = WorldBank()
        payload = wb.fetch_indicator(
            indicator=indicator,
            country=country,
            date=date,
            per_page=args.per_page,
            all_pages=True,
        )
        # Normalize an entire payload; returns list[NormalizedDoc]
        docs = wb.normalize(payload, indicator)

    # elif args.connector == "crawl":
    #     c = SimpleCrawler()
    #     url = args.url or "https://example.com"
    #     payloads = c.fetch(url=url)
    #     docs = [c.normalize(p) for p in payloads]
    #
    # elif args.connector == "pdf":
    #     if not args.pdf:
    #         ap.error("--pdf path is required for pdf connector")
    #     p = PdfIntake()
    #     payloads = p.fetch(path=args.pdf)
    #     docs = [p.normalize(x) for x in payloads]

    else:
        ap.error(f"Unknown connector: {args.connector}")
        return

    # Deduplicate vs state using content_hash (computed by connectors)
    for nd in docs:
        core = getattr(nd, "core", None)
        if core is None:
            # Fallback: assume nd is already a dict-like with 'core'
            core = nd["core"]
        sid = core.source_id if hasattr(core, "source_id") else core.get("source_id")
        chash = (core.content_hash if hasattr(core, "content_hash") else core.get("content_hash")) or ""

        if state.get(sid) == chash:
            continue  # unchanged
        changed.append(nd)
        state[sid] = chash

    if changed:
        append_docs(changed)
        save_state(state)
        print(f"ingested/updated {len(changed)} docs → {DOCS_PATH}")
    else:
        print("no changes detected; nothing to write")


if __name__ == "__main__":
    main()
