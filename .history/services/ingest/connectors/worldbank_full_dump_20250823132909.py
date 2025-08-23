# services/ingest/connectors/worldbank_full_dump.py
import os, time
import requests
from typing import Dict, Any, List, Iterable, Tuple
from pymongo import MongoClient, UpdateOne, ASCENDING
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv
from datetime import datetime

# ==== Config (env-tunable) ====
DATE_RANGE = os.getenv("WB_DATE_RANGE", "1960:2025")   # years to pull
PER_PAGE   = int(os.getenv("WB_PER_PAGE", "1000"))     # WB max 1000
SLEEP_S    = float(os.getenv("WB_SLEEP_S", "0.25"))    # politeness
BASE       = "https://api.worldbank.org/v2"
TIMEOUT    = 60

# Load env + Mongo (local recommended: mongodb://127.0.0.1:27017)
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI missing in .env")
client = MongoClient(MONGO_URI)
db  = client["worldbank_raw"]

# Collections
col_obs        = db["observations"]
col_sources    = db["sources"]
col_countries  = db["countries"]
col_indicators = db["indicators"]
col_state      = db["ingest_state"]

# Indexes (idempotent)
col_obs.create_index([("indicator.id", ASCENDING), ("country.id", ASCENDING), ("date", ASCENDING)])
col_obs.create_index([("source", ASCENDING)])
col_indicators.create_index([("id", ASCENDING)], unique=True)
col_countries.create_index([("id", ASCENDING)], unique=True)
col_sources.create_index([("id", ASCENDING)], unique=True)
# track progress per (indicator_id, source_id)
col_state.create_index([("indicator_id", ASCENDING), ("source_id", ASCENDING)], unique=True)

# Optional limiter per source for testing (0 = no limit)
LIMIT_PER_SOURCE = int(os.getenv("WB_LIMIT_PER_SOURCE", "0"))

# ------------- HTTP helpers -------------
def fetch(url: str) -> List[Any]:
    """Fetch a WB endpoint (JSON) with retry."""
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            # WB success payloads are lists [meta, rows]
            return data
        except Exception as e:
            wait = 2 ** attempt
            print(f"[retry {attempt+1}] {e} -> sleep {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after retries: {url}")

def paged_with_progress(endpoint: str) -> Iterable[Tuple[List[Dict[str, Any]], int, int]]:
    """Yield (rows, page, pages) for paged endpoints."""
    page1_url = f"{endpoint}&per_page={PER_PAGE}&page=1"
    data = fetch(page1_url)
    if not isinstance(data, list) or len(data) < 2:
        yield [], 1, 1
        return
    meta, rows = data[0], data[1] or []
    pages = int(meta.get("pages") or 1)
    yield rows, 1, pages
    for page in range(2, pages + 1):
        time.sleep(SLEEP_S)
        data = fetch(f"{endpoint}&per_page={PER_PAGE}&page={page}")
        rows = data[1] if isinstance(data, list) and len(data) >= 2 else []
        yield (rows or []), page, pages

# ------------- Mongo helpers -------------
def upsert_many(col, ops: List[UpdateOne]):
    if not ops:
        return
    try:
        col.bulk_write(ops, ordered=False)
    except BulkWriteError as e:
        # Ignore duplicates/races; raise other errors
        write_errors = e.details.get("writeErrors", [])
        non_dupes = [we for we in write_errors if we.get("code") not in (11000, 11001)]
        if non_dupes:
            raise

# ------------- Metadata -------------
def store_sources():
    print("Fetching sources…")
    endpoint = f"{BASE}/sources?format=json"
    batch = []
    for rows, _, _ in paged_with_progress(endpoint):
        for s in rows or []:
            batch.append(UpdateOne({"id": s.get("id")}, {"$set": s}, upsert=True))
            if len(batch) >= 1000:
                upsert_many(col_sources, batch); batch = []
    upsert_many(col_sources, batch)
    print(f"Sources upserted: {col_sources.count_documents({})}")

def store_countries():
    print("Fetching countries…")
    endpoint = f"{BASE}/country?format=json"
    batch = []
    for rows, _, _ in paged_with_progress(endpoint):
        for c in rows or []:
            doc = {
                "id": c.get("id"),
                "name": c.get("name"),
                "region": (c.get("region") or {}).get("id"),
                "incomeLevel": (c.get("incomeLevel") or {}).get("id"),
                "lendingType": (c.get("lendingType") or {}).get("id"),
                "iso2Code": c.get("iso2Code"),
                "capitalCity": c.get("capitalCity"),
                "longitude": c.get("longitude"),
                "latitude": c.get("latitude"),
                "raw": c
            }
            batch.append(UpdateOne({"id": doc["id"]}, {"$set": doc}, upsert=True))
            if len(batch) >= 1000:
                upsert_many(col_countries, batch); batch = []
    upsert_many(col_countries, batch)
    print(f"Countries upserted: {col_countries.count_documents({})}")

def store_indicators_for_source(source_id: int):
    print(f"Fetching indicators for source {source_id}…")
    endpoint = f"{BASE}/sources/{source_id}/indicators?format=json"
    batch, total = [], 0
    for rows, _, _ in paged_with_progress(endpoint):
        for ind in rows or []:
            doc = {**ind, "source_id": source_id}
            batch.append(UpdateOne({"id": ind.get("id")}, {"$set": doc}, upsert=True))
            total += 1
            if len(batch) >= 1000:
                upsert_many(col_indicators, batch); batch = []
    upsert_many(col_indicators, batch)
    print(f"Indicators upserted for source {source_id}: {total}")

# ------------- Observations -------------
def obs_id(indicator_id: str, country_id: str, date: str) -> str:
    return f"{indicator_id}|{country_id}|{date}"

def dump_indicator_all_countries(indicator_id: str, source_id: int, g_idx: int, g_total: int):
    # Resume check
    if col_state.find_one({"indicator_id": indicator_id, "source_id": source_id, "status": "done"}):
        if g_idx % 50 == 0:
            print(f"[{g_idx}/{g_total}] {indicator_id} -> already done")
        return

    title = f"[{g_idx}/{g_total}] {indicator_id}"
    endpoint = f"{BASE}/country/all/indicator/{indicator_id}?format=json&date={DATE_RANGE}"
    ops, count = [], 0
    first_page_url = f"{endpoint}&per_page={PER_PAGE}&page=1"

    for rows, page, pages in paged_with_progress(endpoint):
        for r in rows:
            if not r or "indicator" not in r or "country" not in r:
                continue
            ind_id  = (r.get("indicator") or {}).get("id") or indicator_id
            c       = r.get("country") or {}
            c_id    = c.get("id") or ""
            c_name  = c.get("value") or ""
            iso3    = r.get("countryiso3code") or c_id
            date    = str(r.get("date") or "")
            key     = obs_id(ind_id, c_id, date)
            doc = {
                "_id": key,
                "source": source_id,
                "indicator": {"id": ind_id, "name": (r.get("indicator") or {}).get("value")},
                "country": {"id": c_id, "iso3": iso3, "name": c_name},
                "date": date,
                "value": r.get("value"),
                "unit": r.get("unit"),
                "obs_status": r.get("obs_status"),
                "decimal": r.get("decimal"),
                "meta": {"api_first_page": first_page_url}
            }
            ops.append(UpdateOne({"_id": key}, {"$set": doc}, upsert=True))
            if len(ops) >= 1000:
                upsert_many(col_obs, ops); count += len(ops); ops = []
        # progress every 10 pages or last page
        if pages <= 10 or page % 10 == 0 or page == pages:
            pct = (page / pages) * 100 if pages else 100.0
            print(f"{title} pages {page}/{pages} ({pct:.1f}%), rows so far: {count}")
        time.sleep(SLEEP_S)

    if ops:
        upsert_many(col_obs, ops); count += len(ops)

    col_state.update_one(
        {"indicator_id": indicator_id, "source_id": source_id},
        {"$set": {
            "indicator_id": indicator_id,
            "source_id": source_id,
            "status": "done",
            "rows_upserted": count,
            "finished_at": datetime.utcnow()
        }},
        upsert=True
    )
    print(f"{title} -> upserted rows: {count}")

# ------------- Orchestrator -------------
def run(full_sources: bool = True):
    # 1) metadata
    store_sources()
    store_countries()

    # 2) source list
    src_cursor = col_sources.find({}, {"id": 1})
    source_ids = [s["id"] for s in src_cursor] if full_sources else [2]  # 2 = WDI
    print(f"Processing sources: {source_ids}")

    # 3) indicators metadata per source
    for sid in source_ids:
        try:
            store_indicators_for_source(sid)
        except Exception as e:
            print(f"Failed indicators for source {sid}: {e}")

    # 4) global worklist (for overall progress)
    worklist: List[Tuple[int, str]] = []
    for sid in source_ids:
        ids = list(col_indicators.find({"source_id": sid}, {"id": 1}).sort([("id", ASCENDING)]))
        if LIMIT_PER_SOURCE:
            ids = ids[:LIMIT_PER_SOURCE]
        for x in ids:
            worklist.append((sid, x["id"]))

    total = len(worklist)
    done  = col_state.count_documents({"status": "done", "source_id": {"$in": source_ids}})
    print(f"Overall indicators: {done}/{total} ({(done/total*100 if total else 100):.1f}%)")

    # 5) dump loop
    for idx, (sid, ind_id) in enumerate(worklist, start=1):
        try:
            if col_state.find_one({"indicator_id": ind_id, "source_id": sid, "status": "done"}):
                if idx % 50 == 0:
                    print(f"[{idx}/{total}] (skipped already done)")
                continue
            dump_indicator_all_countries(ind_id, sid, g_idx=idx, g_total=total)
            done += 1
            if idx % 10 == 0 or idx == total:
                print(f"== Progress: {done}/{total} ({done/total*100:.1f}%) | obs: ~{col_obs.estimated_document_count():,} ==")
        except Exception as e:
            print(f"[{sid}:{idx}/{total}] {ind_id} -> {e}")

if __name__ == "__main__":
    run(full_sources=True)
