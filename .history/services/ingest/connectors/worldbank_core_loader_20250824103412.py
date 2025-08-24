# services/ingest/connectors/worldbank_core_loader.py
import os, time, threading
from typing import Any, Dict, List, Iterable, Set
import requests
from pymongo import MongoClient, UpdateOne, ASCENDING
from pymongo.errors import BulkWriteError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Config (override via .env) ---
INDICATORS = [
    # Economy
    "NY.GDP.PCAP.KD","NY.GDP.MKTP.KD.ZG","NE.EXP.GNFS.ZS","FP.CPI.TOTL.ZG",
    # Labor
    "SL.UEM.TOTL.ZS","SL.TLF.CACT.ZS","SL.EMP.TOTL.SP.ZS",
    # Demographics & Health
    "SP.POP.TOTL","SP.DYN.LE00.IN","SH.XPD.CHEX.PC.CD","SH.IMM.MEAS.ZS",
    # Education
    "SE.SEC.ENRR","SE.TER.ENRR","SE.ADT.LITR.ZS",
    # ICT / Digital
    "IT.NET.USER.ZS","IT.CEL.SETS.P2","IT.NET.BBND.P2","BX.GSR.CCIS.CD",
    # Environment & Energy
    "EN.ATM.CO2E.PC","EN.ATM.PM25.MC.M3","EG.ELC.ACCS.ZS","EG.ELC.RNEW.ZS","EG.USE.PCAP.KG.OE",
    # Business / Finance / Innovation / Trade
    "IC.LGL.CRED.XQ","IC.BUS.NDNS.ZS","FS.AST.PRVT.GD.ZS","TX.VAL.TECH.MF.ZS","IP.JRN.ARTC.SC",
    # Affordability & Safety
    "PA.NUS.PPPC.RF","SH.STA.HOMIC.ZS",
    # Urbanization
    "SP.URB.TOTL.IN.ZS",
]

BASE = "https://api.worldbank.org/v2"
DATE_RANGE = os.getenv("WB_DATE_RANGE", "1990:2025")
PER_PAGE   = int(os.getenv("WB_PER_PAGE", "1000"))
TIMEOUT_S  = int(os.getenv("WB_TIMEOUT_S", "60"))
SLEEP_S    = float(os.getenv("WB_SLEEP_S", "0.05"))
WORKERS    = int(os.getenv("WB_CORE_WORKERS", "8"))
COUNTRIES_ONLY = os.getenv("WB_COUNTRIES_ONLY", "1") not in ("0","false","False")

# --- Mongo ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise SystemExit("MONGO_URI missing in .env")
client = MongoClient(MONGO_URI)
db  = client["worldbank_core"]
col_obs = db["observations"]
col_countries = db["countries"]
col_indicators = db["indicators"]

print(f"Target DB: {db.name} | indicators: {len(INDICATORS)} | range: {DATE_RANGE} | workers: {WORKERS}")

# --- HTTP helpers ---
def new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "wb-core-loader/1.0"})
    return s

def fetch_json(sess: requests.Session, url: str) -> Any:
    for attempt in range(5):
        try:
            r = sess.get(url, timeout=TIMEOUT_S)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"[retry {attempt+1}] {e} -> sleep {wait}s | {url}")
            time.sleep(wait)
    raise RuntimeError(f"Failed after retries: {url}")

def paged(sess: requests.Session, endpoint: str) -> Iterable[List[Dict[str, Any]]]:
    page = 1
    url  = f"{endpoint}&per_page={PER_PAGE}&page={page}"
    data = fetch_json(sess, url)
    if not isinstance(data, list) or len(data) < 2:
        yield []
        return
    meta, rows = data[0], data[1] or []
    pages = int(meta.get("pages") or 1)
    yield rows
    for page in range(2, pages + 1):
        time.sleep(SLEEP_S)
        data = fetch_json(sess, f"{endpoint}&per_page={PER_PAGE}&page={page}")
        more = data[1] if isinstance(data, list) and len(data) >= 2 else []
        yield more or []

# --- Country filter (exclude aggregates) ---
def get_real_country_ids(sess: requests.Session) -> Set[str]:
    keep: Set[str] = set()
    endpoint = f"{BASE}/country?format=json"
    for rows in paged(sess, endpoint):
        for c in rows:
            if not c: continue
            region_id = (c.get("region") or {}).get("id")
            iso3 = (c.get("id") or "").upper()      # WB country "id" is ISO-3 (e.g., AUT)
            if region_id and region_id != "NA" and iso3:
                keep.add(iso3)
                col_countries.update_one(
                    {"id": iso3},
                    {"$set": {
                        "id": iso3,
                        "name": c.get("name"),
                        "region": region_id,
                        "incomeLevel": (c.get("incomeLevel") or {}).get("id"),
                        "iso2Code": (c.get("iso2Code") or "").upper(),
                        "capitalCity": c.get("capitalCity"),
                    }},
                    upsert=True
                )
    print(f"Countries (non-aggregates): {len(keep)}")
    return keep

# --- Upsert helper ---
def upsert_many(col, ops: List[UpdateOne]):
    if not ops: return 0
    try:
        res = col.bulk_write(ops, ordered=False, bypass_document_validation=True)
        return (res.upserted_count or 0) + (res.modified_count or 0)
    except BulkWriteError as e:
        write_errors = [we for we in e.details.get("writeErrors", []) if we.get("code") not in (11000,11001)]
        if write_errors:
            raise
        return 0

# --- Indicator worker ---
_print_lock = threading.Lock()
def process_indicator(ind_id: str, countries_keep: Set[str]) -> int:
    sess = new_session()
    endpoint = f"{BASE}/country/all/indicator/{ind_id}?format=json&date={DATE_RANGE}"
    first_page = f"{endpoint}&per_page={PER_PAGE}&page=1"

    ops: List[UpdateOne] = []
    total = 0
    for rows in paged(sess, endpoint):
        for r in rows:
            if not r or "country" not in r or "indicator" not in r:
                continue
            c = r.get("country") or {}
            iso2 = (c.get("id") or "").upper()                 # sometimes ISO-2 in other WB endpoints
            iso3 = (r.get("countryiso3code") or iso2).upper()  # prefer ISO-3; fallback to iso2 if missing

            # filter using ISO-3 list
            if COUNTRIES_ONLY and iso3 and iso3 not in countries_keep:
                continue

            date = str(r.get("date") or "")
            doc = {
                "_id": f"{ind_id}|{iso3}|{date}",
                "indicator": {"id": ind_id, "name": (r.get("indicator") or {}).get("value")},
                "country":   {"id": iso3, "iso3": iso3, "iso2": iso2, "name": c.get("value")},
                "date": date,
                "value": r.get("value"),
                "unit": r.get("unit"),
                "obs_status": r.get("obs_status"),
                "decimal": r.get("decimal"),
                "meta": {"api_first_page": first_page}
            }
            ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))
            if len(ops) >= 5000:
                total += upsert_many(col_obs, ops); ops = []
        time.sleep(SLEEP_S)
    if ops:
        total += upsert_many(col_obs, ops)

    # ensure minimal indicator meta doc exists
    col_indicators.update_one({"id": ind_id}, {"$setOnInsert": {"id": ind_id}}, upsert=True)

    with _print_lock:
        print(f"[OK] {ind_id} -> upserted ~{total} rows")
    return total

def main():
    sess = new_session()
    keep = get_real_country_ids(sess) if COUNTRIES_ONLY else set()

    started = time.time()
    totals = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(process_indicator, ind, keep) for ind in INDICATORS]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                totals += fut.result()
            except Exception as e:
                with _print_lock:
                    print(f"[ERR] {e}")

            if i % 5 == 0:
                elapsed = time.time() - started
                rate = totals / elapsed if elapsed > 0 else 0
                with _print_lock:
                    print(f"== Progress: {i}/{len(INDICATORS)} | rows ~{totals:,} | {rate:,.0f} rows/s ==")

    # Indexes (light dataset)
    col_obs.create_index([("indicator.id", ASCENDING), ("country.id", ASCENDING), ("date", ASCENDING)])
    col_obs.create_index([("country.id", ASCENDING), ("date", ASCENDING)])
    print("Done.")
    print(col_obs.estimated_document_count(), "documents in worldbank_core.observations")

if __name__ == "__main__":
    main()
