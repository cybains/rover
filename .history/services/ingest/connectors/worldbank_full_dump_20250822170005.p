# services/ingest/connectors/worldbank_full_dump.py
import os, time, math, hashlib
import requests
from typing import Dict, Any, List, Iterable
from pymongo import MongoClient, UpdateOne, ASCENDING
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv

# ==== Config ====
DATE_RANGE = os.getenv("WB_DATE_RANGE", "1960:2025")   # wide range; adjust if needed
PER_PAGE   = int(os.getenv("WB_PER_PAGE", "1000"))     # WB max page size
SLEEP_S    = float(os.getenv("WB_SLEEP_S", "0.25"))    # politeness between HTTP calls
BASE       = "https://api.worldbank.org/v2"
TIMEOUT    = 60

# Load env + Mongo
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI missing in .env")

client = MongoClient(MONGO_URI)
db  = client["worldbank_raw"]
col_obs       = db["observations"]
col_sources   = db["sources"]
col_countries = db["countries"]
col_indicators= db["indicators"]
col_state     = db["ingest_state"]

# Indexes (run once / idempotent)
col_obs.create_index([("indicator.id", ASCENDING), ("country.id", ASCENDING), ("date", ASCENDING)])
col_obs.create_index([("source", ASCENDING)])
col_obs.create_index([("_id", ASCENDING)], unique=True)
col_indicators.create_index([("id", ASCENDING)], unique=True)
col_countries.create_index([("id", ASCENDING)], unique=True)
col_sources.create_index([("id", ASCENDING)], unique=True)
col_state.create_index([("indicator_id", ASCENDING)], unique=True)

def fetch(url: str) -> List[Any]:
    """Fetch a WB endpoint (JSON) with basic retry + pagination support if first page URL is passed."""
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list) or len(data) < 2:
                return data  # message payloads (errors) or odd endpoints
            return data
        except Exception as e:
            wait = 2 ** attempt
            print(f"[retry {attempt+1}] {e} -> sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed after retries: {url}")

def paged(endpoint: str) -> Iterable[List[Dict[str, Any]]]:
    """Yield rows across all pages for an endpoint that supports ?page=&per_page=."""
    page = 1
    url  = f"{endpoint}&per_page={PER_PAGE}&page={page}"
    data = fetch(url)
    if not isinstance(data, list) or len(data) < 2:
        yield []
        return
    meta, rows = data[0], data[1] or []
    pages = int(meta.get("pages") or 1)
    yield rows
    for page in range(2, pages + 1):
        time.sleep(SLEEP_S)
        data = fetch(f"{endpoint}&per_page={PER_PAGE}&page={page}")
        more = data[1] if isinstance(data, list) and len(data) >= 2 else []
        yield more or []

def upsert_many(col, ops: List[UpdateOne]):
    if not ops: 
        return
    try:
        col.bulk_write(ops, ordered=False)
    except BulkWriteError as e:
        # Ignore duplicate/upsert races; surface other errors
        write_errors = e.details.get("writeErrors", [])
        non_dupes = [we for we in write_errors if we.get("code") not in (11000, 11001)]
        if non_dupes:
            raise

def store_sources():
    print("Fetching sources…")
    endpoint = f"{BASE}/sources?format=json"
    batch = []
    for rows in paged(endpoint):
        for s in rows:
            if not s: continue
            batch.append(UpdateOne({"id": s.get("id")}, {"$set": s}, upsert=True))
        if len(batch) >= 1000:
            upsert_many(col_sources, batch); batch = []
    upsert_many(col_sources, batch)
    print(f"Sources upserted: {col_sources.count_documents({})}")

def store_countries():
    print("Fetching countries…")
    endpoint = f"{BASE}/country?format=json"
    batch = []
    for rows in paged(endpoint):
        for c in rows:
            if not c: continue
            doc = {
                "id": c.get("id"),
                "name": c.get("name"),
                "region": c.get("region", {}).get("id"),
                "incomeLevel": c.get("incomeLevel", {}).get("id"),
                "lendingType": c.get("lendingType", {}).get("id"),
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
    batch = []
    total = 0
    for rows in paged(endpoint):
        for ind in rows:
            if not ind: continue
            doc = {**ind, "source_id": source_id}
            batch.append(UpdateOne({"id": ind.get("id")}, {"$set": doc}, upsert=True))
            total += 1
        if len(batch) >= 1000:
            upsert_many(col_indicators, batch); batch = []
    upsert_many(col_indicators, batch)
    print(f"Indicators upserted for source {source_id}: {total}")

def obs_key(indicator_id: str, country_id: str, date: str) -> str:
    return f"{indicator_id}|{country_id}|{date}"

def dump_indicator_all_countries(indicator_id: str, source_id: int):
    print(f"→ Dumping indicator {indicator_id} (source {source_id})")
    # state check (skip if already fully processed)
    st = col_state.find_one({"indicator_id": indicator_id})
    if st and st.get("status") == "done":
        print(f"  already done, skipping")
        return

    endpoint = f"{BASE}/country/all/indicator/{indicator_id}?format=json&date={DATE_RANGE}"
    ops = []
    count = 0
    first_page_url = f"{endpoint}&per_page={PER_PAGE}&page=1"

    for rows in paged(endpoint):
        for r in rows:
            if not r or "indicator" not in r or "country" not in r:
                continue
            ind_id   = (r.get("indicator") or {}).get("id") or indicator_id
            country  = r.get("country") or {}
            c_id     = country.get("id") or ""
            c_name   = country.get("value") or ""
            iso3     = r.get("countryiso3code") or c_id
            date     = str(r.get("date") or "")
            value    = r.get("value")
            unit     = r.get("unit")
            obs_stat = r.get("obs_status")
            deci     = r.get("decimal")

            key = obs_key(ind_id, c_id, date)
            doc = {
                "_id": key,
                "source": source_id,
                "indicator": {"id": ind_id, "name": (r.get("indicator") or {}).get("value")},
                "country": {"id": c_id, "iso3": iso3, "name": c_name},
                "date": date,      # keep as string to avoid weird 'n.d.' cases
                "value": value,
                "unit": unit,
                "obs_status": obs_stat,
                "decimal": deci,
                "meta": {"api_first_page": first_page_url}
            }
            ops.append(UpdateOne({"_id": key}, {"$set": doc}, upsert=True))
            if len(ops) >= 1000:
                upsert_many(col_obs, ops); count += len(ops); ops = []
        time.sleep(SLEEP_S)

    if ops:
        upsert_many(col_obs, ops); count += len(ops)

    col_state.update_one(
        {"indicator_id": indicator_id},
        {"$set": {"indicator_id": indicator_id, "status": "done", "rows_upserted": count, "source_id": source_id}},
        upsert=True
    )
    print(f"  upserted rows: {count}")

def run(full_sources: bool = True):
    # 1) metadata
    store_sources()
    store_countries()

    # Pick sources
    src_cursor = col_sources.find({}, {"id": 1})
    source_ids = [s["id"] for s in src_cursor] if full_sources else [2]  # 2 = WDI
    print(f"Processing sources: {source_ids}")

    # 2) indicators metadata per source
    for sid in source_ids:
        try:
            store_indicators_for_source(sid)
        except Exception as e:
            print(f"Failed indicators for source {sid}: {e}")

    # 3) dump observations for each indicator (by source)
    # You can limit here by env: WB_LIMIT_PER_SOURCE
    limit = int(os.getenv("WB_LIMIT_PER_SOURCE", "0"))  # 0 = no limit
    for sid in source_ids:
        inds = list(col_indicators.find({"source_id": sid}, {"id": 1}).sort([("id", ASCENDING)]))
        if limit:
            inds = inds[:limit]
        print(f"Source {sid}: {len(inds)} indicators")
        for i, ind in enumerate(inds, 1):
            ind_id = ind["id"]
            try:
                dump_indicator_all_countries(ind_id, sid)
            except Exception as e:
                print(f"[{sid}:{i}/{len(inds)}] {ind_id} -> {e}")

if __name__ == "__main__":
    # NOTE: This is massive. To start with only WDI set WB_LIMIT_PER_SOURCE=20 in .env for a dry run.
    run(full_sources=True)
