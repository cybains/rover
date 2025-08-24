import os, time, statistics
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne, ASCENDING
from pymongo.errors import BulkWriteError

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI: raise SystemExit("MONGO_URI missing in .env")
client = MongoClient(MONGO_URI)
db = client["worldbank_core"]
col_feat = db["features"]
col_ctry = db["countries"]
col_ps   = db["persona_scores"]

# ---------------- Persona configs ----------------
PERSONAS: Dict[str, Dict[str, List[str]]] = {
    "job_seeker": {
        "employment_health": ["SL.UEM.TOTL.ZS"],
        "participation_skills": ["SL.TLF.CACT.ZS", "SE.TER.ENRR"],
        "momentum": ["NY.GDP.MKTP.KD.ZG", "NE.EXP.GNFS.ZS"],
        "digital_access": ["IT.NET.USER.ZS", "IT.CEL.SETS.P2"],
    },
    "entrepreneur": {
        "regulatory_legal": ["RL.EST", "IC.BUS.NDNS.ZS"],
        "access_finance": ["FS.AST.PRVT.GD.ZS", "IC.CRD.PRVT.ZS"],
        "infrastructure_power": ["EG.ELC.ACCS.ZS", "EG.ELC.RNEW.ZS"],
        "innovation_trade": ["TX.VAL.TECH.MF.ZS", "IP.JRN.ARTC.SC"],
    },
    "digital_nomad": {
        "connectivity": ["IT.NET.USER.ZS", "IT.NET.BBND.P2", "IT.CEL.SETS.P2"],
        "afford_stability": ["PA.NUS.PPPC.RF", "FP.CPI.TOTL.ZG"],
        "livability_safety": ["SP.DYN.LE00.IN", "EN.ATM.PM25.MC.M3", "VC.IHR.PSRC.P5"],
    },
    "expat_family": {
        "health": ["SP.DYN.LE00.IN", "SH.XPD.CHEX.PC.CD", "SH.IMM.MEAS"],
        "education": ["SE.SEC.ENRR", "SE.TER.ENRR", "SE.ADT.LITR.ZS"],
        "safety_env": ["VC.IHR.PSRC.P5", "EN.ATM.PM25.MC.M3", "EN.ATM.CO2E.PC"],
    },
}

# ---------------- Recency window ----------------
MAX_LAG_YEARS = 3  # default lookback window
# per-indicator overrides (slower-moving series get a wider window)
IND_MAX_LAG = {
    "RL.EST": 3,                 # WGI annual, sometimes 1y lag
    "VC.IHR.PSRC.P5": 7,         # homicide can lag
    "SE.ADT.LITR.ZS": 15,        # literacy updates infrequent
    "SH.IMM.MEAS": 5,
    "SH.XPD.CHEX.PC.CD": 3,
    "IC.BUS.NDNS.ZS": 7,         # new business density
    "IT.NET.BBND.P2": 3,
    "EG.ELC.ACCS.ZS": 5,
    "EG.ELC.RNEW.ZS": 5,
    "EN.ATM.CO2E.PC": 3,
    "EN.ATM.PM25.MC.M3": 3,
    "FS.AST.PRVT.GD.ZS": 5,
    "IC.CRD.PRVT.ZS": 5,
    "TX.VAL.TECH.MF.ZS": 5,
    "IP.JRN.ARTC.SC": 5,
    "PA.NUS.PPPC.RF": 3,
}

MIN_COVERAGE = 0.60  # persona-level coverage threshold

def avg(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(statistics.fmean(xs)) if xs else None

def upsert_many(ops: List[UpdateOne]):
    if not ops: return 0
    try:
        res = col_ps.bulk_write(ops, ordered=False, bypass_document_validation=True)
        return (res.upserted_count or 0) + (res.modified_count or 0)
    except BulkWriteError as e:
        non_dupes = [we for we in e.details.get("writeErrors", []) if we.get("code") not in (11000,11001)]
        if non_dupes: raise
        return 0

# ----------- helpers to select latest within window -----------
def select_within_window(ts: Dict[int, float], target_year: int, code: str) -> Tuple[Optional[int], Optional[float]]:
    """Pick the most recent year <= target_year within allowed lag; return (year_used, value)."""
    if not ts: return (None, None)
    max_lag = IND_MAX_LAG.get(code, MAX_LAG_YEARS)
    best_y = None
    best_v = None
    for y in sorted(ts.keys(), reverse=True):
        if y <= target_year and (target_year - y) <= max_lag:
            best_y = y
            best_v = ts[y]
            break
    return (best_y, best_v)

def build_for_country(iso3: str) -> List[UpdateOne]:
    # 1) load all needed features for this country
    needed_codes = sorted({code for p in PERSONAS.values() for codes in p.values() for code in codes})
    cur = col_feat.find(
        {"country": iso3, "indicator": {"$in": needed_codes}},
        {"_id":0, "indicator":1, "year":1, "pctl_world":1}
    )
    # series_by_code: code -> {year: pctl}
    series_by_code: Dict[str, Dict[int, float]] = defaultdict(dict)
    years_all = set()
    for d in cur:
        code = d["indicator"]; y = int(d["year"]); v = d.get("pctl_world")
        if v is None: continue
        series_by_code[code][y] = float(v)
        years_all.add(y)

    if not years_all:
        return []

    # We'll compute for every year that appears in any needed code
    years_sorted = sorted(years_all)

    ops: List[UpdateOne] = []
    for y in years_sorted:
        personas_out = {}
        for persona_name, pillars in PERSONAS.items():
            pillar_out = {}
            pillar_scores = []
            total_needed = 0
            total_present = 0
            max_lag_used = 0

            for pillar_name, codes in pillars.items():
                ind_map = {}
                vals = []
                for code in codes:
                    year_used, val = select_within_window(series_by_code.get(code, {}), y, code)
                    ind_map[code] = {
                        "score": val,            # 0..100 percentile (polarity-adjusted)
                        "year_used": year_used,  # may be < y
                        "lag": (y - year_used) if year_used is not None else None
                    }
                    if val is not None:
                        vals.append(val)
                        total_present += 1
                        if year_used is not None:
                            max_lag_used = max(max_lag_used, y - year_used)
                    total_needed += 1

                pillar_score = avg(vals)
                coverage = len([1 for v in vals if v is not None]) / len(codes) if codes else 0.0
                pillar_out[pillar_name] = {
                    "score": pillar_score,
                    "coverage": coverage,
                    "n_present": len(vals),
                    "n_needed": len(codes),
                    "indicators": ind_map
                }
                if pillar_score is not None:
                    pillar_scores.append(pillar_score)

            persona_score = avg(pillar_scores)
            coverage_overall = (total_present / total_needed) if total_needed else 0.0
            personas_out[persona_name] = {
                "score": persona_score,
                "coverage": coverage_overall,
                "low_coverage": coverage_overall < MIN_COVERAGE,
                "max_lag_used": max_lag_used,
                "pillars": pillar_out
            }

        doc = {
            "_id": f"{iso3}|{y}",
            "country": iso3,
            "year": y,
            "personas": personas_out,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "params": {
                "max_lag_default": MAX_LAG_YEARS,
                "ind_max_lag": IND_MAX_LAG,
                "min_coverage": MIN_COVERAGE
            }
        }
        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))
    return ops

def main():
    t0 = time.time()
    ops: List[UpdateOne] = []
    countries = [c["id"] for c in col_ctry.find({}, {"id":1})]
    for i, iso3 in enumerate(countries, 1):
        ops.extend(build_for_country(iso3))
        if len(ops) >= 5000:
            upsert_many(ops); ops = []
        if i % 25 == 0:
            print(f"[persona] countries processed: {i}/{len(countries)}")
    if ops: upsert_many(ops)
    col_ps.create_index([("country", ASCENDING), ("year", ASCENDING)], name="country_year")
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
