# services/ingest/connectors/country_views_materializer.py
import os, time, math
from typing import Dict, List, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from pymongo import MongoClient, ASCENDING, UpdateOne

HEADLINE = [
    "NY.GDP.PCAP.KD","FP.CPI.TOTL.ZG","NE.EXP.GNFS.ZS","SL.UEM.TOTL.ZS",
    "SL.TLF.CACT.ZS","SP.POP.TOTL","SP.DYN.LE00.IN","SE.TER.ENRR",
    "SE.ADT.LITR.ZS","IT.NET.USER.ZS","IT.CEL.SETS.P2","EN.ATM.CO2E.PC"
]

ABS_IS_PP = {
    "FP.CPI.TOTL.ZG","SL.UEM.TOTL.ZS","NE.EXP.GNFS.ZS","SL.TLF.CACT.ZS",
    "SE.SEC.ENRR","SE.TER.ENRR","SE.ADT.LITR.ZS","IT.NET.USER.ZS",
    "EG.ELC.ACCS.ZS","EG.ELC.RNEW.ZS","IT.NET.BBND.P2","SP.URB.TOTL.IN.ZS"
}

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI: raise SystemExit("MONGO_URI missing")

cli = MongoClient(MONGO_URI)
db  = cli["worldbank_core"]
feat = db["features"]
obs  = db["observations"]
ctry = db["countries"]
views= db["country_views"]

def percentile(val: float, arr: List[float]) -> float:
    if not arr: return float("nan")
    # percent <= val
    n = len(arr)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] <= val: lo = mid + 1
        else: hi = mid
    return 100.0 * lo / n

def get_country_meta() -> Dict[str, Dict]:
    out = {}
    for d in ctry.find({}, {"_id":0}):
        out[d["id"]] = {
            "id": d["id"],
            "name": d.get("name"),
            "region": d.get("region"),
            "income": d.get("incomeLevel"),
            "iso2":  d.get("iso2Code")
        }
    return out

def latest_feature(ind: str, co: str):
    q = {"indicator": ind, "country": co}
    doc = feat.find(q, {"_id":0}).sort("year",-1).limit(1)
    return next(doc, None)

def last_n_series(ind: str, co: str, n: int = 10):
    cur = feat.find({"indicator": ind, "country": co}, {"_id":0,"year":1,"value":1}) \
              .sort("year",-1).limit(n)
    arr = [{"year": d["year"], "value": d.get("value")} for d in cur]
    return list(reversed(arr))

def region_income_peer_sets(ind: str, year: int, region: str, income: str):
    # Pull value_clip of all countries for this indicator-year
    cur = feat.find({"indicator": ind, "year": year}, {"_id":0,"country":1,"value_clip":1})
    vals_region, vals_income = [], []
    # Build lookup for region/income from cached meta
    for d in cur:
        co = d["country"]
        v  = d.get("value_clip")
        if v is None: continue
        meta = CTRY_META.get(co)
        if not meta: continue
        if region and meta.get("region")==region: vals_region.append(v)
        if income and meta.get("income")==income: vals_income.append(v)
    vals_region.sort(); vals_income.sort()
    return vals_region, vals_income

def build_views():
    ops = []
    countries = list(ctry.find({}, {"id":1}))
    t0 = time.time()
    for i, cdoc in enumerate(countries, 1):
        co = cdoc["id"]
        meta = CTRY_META.get(co, {})
        region = meta.get("region"); income = meta.get("income")

        entries = []
        for ind in HEADLINE:
            lf = latest_feature(ind, co)
            if not lf: 
                entries.append({"code": ind, "status":"no_data"})
                continue

            year = lf["year"]; val = lf.get("value")
            yoy_pct = lf.get("yoy_pct"); yoy_abs = lf.get("yoy_abs")
            world_pctl = lf.get("pctl_world")
            # compute region/income percentile on-the-fly
            rvals, ivals = region_income_peer_sets(ind, year, region, income)
            rp = percentile(lf.get("value_clip"), rvals) if rvals else None
            ip = percentile(lf.get("value_clip"), ivals) if ivals else None

            # unit: get from observations (same ind, co, year)
            unit = None
            ob = obs.find_one({"indicator.id": ind, "country.id": co, "date": str(year)}, {"unit":1})
            if ob: unit = ob.get("unit")

            entries.append({
                "code": ind,
                "year": year,
                "value": val,
                "unit": unit,
                "yoy": ({"pp": yoy_abs} if ind in ABS_IS_PP else {"pct": yoy_pct}),
                "pctl": {"world": world_pctl, "region": rp, "income": ip},
                "trend": last_n_series(ind, co, 10)
            })

        doc = {
            "_id": co,
            "country": {"id": co, "name": meta.get("name"), "region": region, "income": income, "iso2": meta.get("iso2")},
            "headlines": entries,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        ops.append(UpdateOne({"_id": co}, {"$set": doc}, upsert=True))
        if len(ops) >= 500:
            views.bulk_write(ops, ordered=False); ops = []
        if i % 25 == 0:
            print(f"[views] {i}/{len(countries)}")

    if ops: views.bulk_write(ops, ordered=False)
    views.create_index([("country.id", ASCENDING)], name="country_id")
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    CTRY_META = get_country_meta()
    build_views()
