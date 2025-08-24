# services/ingest/connectors/persona_scores_materializer.py
import os, time, statistics
from typing import Dict, List, Optional
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

# --- Persona configs (equal pillar weights; equal within pillar). ---
# Note: We replaced deprecated Doing Business indicator with stable proxies available in your core:
#   Regulatory/legal -> use RL.EST + IC.BUS.NDNS.ZS
#   Access to finance -> FS.AST.PRVT.GD.ZS + IC.CRD.PRVT.ZS
PERSONAS: Dict[str, Dict[str, List[str]]] = {
    "job_seeker": {
        "employment_health": ["SL.UEM.TOTL.ZS"],                     # lower-better already handled in pctl_world
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
        "afford_stability": ["PA.NUS.PPPC.RF", "FP.CPI.TOTL.ZG"],    # both lower-better already inverted in features
        "livability_safety": ["SP.DYN.LE00.IN", "EN.ATM.PM25.MC.M3", "VC.IHR.PSRC.P5"],
    },
    "expat_family": {
        "health": ["SP.DYN.LE00.IN", "SH.XPD.CHEX.PC.CD", "SH.IMM.MEAS"],
        "education": ["SE.SEC.ENRR", "SE.TER.ENRR", "SE.ADT.LITR.ZS"],
        "safety_env": ["VC.IHR.PSRC.P5", "EN.ATM.PM25.MC.M3", "EN.ATM.CO2E.PC"],
    },
}

MIN_COVERAGE = 0.60  # overall indicator coverage threshold to flag low_coverage

def avg(xs: List[float]) -> Optional[float]:
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

def build_for_country(iso3: str) -> List[UpdateOne]:
    # Pull all features for this country that are used in any persona
    needed_codes = sorted({code for pillars in PERSONAS.values() for codes in pillars.values() for code in codes})
    cur = col_feat.find(
        {"country": iso3, "indicator": {"$in": needed_codes}},
        {"_id":0, "indicator":1, "year":1, "pctl_world":1}
    )
    by_year: Dict[int, Dict[str, float]] = defaultdict(dict)
    for d in cur:
        by_year[d["year"]][d["indicator"]] = d.get("pctl_world")

    ops: List[UpdateOne] = []
    years = sorted(by_year.keys())
    for y in years:
        year_map = by_year[y]
        personas_out = {}
        for persona_name, pillars in PERSONAS.items():
            pillar_out = {}
            pillar_scores = []
            total_needed = 0
            total_present = 0
            for pillar_name, codes in pillars.items():
                vals = [year_map.get(code) for code in codes if year_map.get(code) is not None]
                cov = len(vals) / len(codes) if codes else 0.0
                score = avg(vals)  # already 0..100
                pillar_out[pillar_name] = {
                    "score": score,
                    "coverage": cov,
                    "n_present": len(vals),
                    "n_needed": len(codes),
                    "indicators": {code: year_map.get(code) for code in codes}
                }
                if score is not None:
                    pillar_scores.append(score)
                total_needed += len(codes)
                total_present += len(vals)

            persona_score = avg(pillar_scores)
            coverage_overall = (total_present / total_needed) if total_needed else 0.0
            personas_out[persona_name] = {
                "score": persona_score,
                "coverage": coverage_overall,
                "low_coverage": coverage_overall < MIN_COVERAGE,
                "pillars": pillar_out
            }

        doc = {
            "_id": f"{iso3}|{y}",
            "country": iso3,
            "year": y,
            "personas": personas_out,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        }
        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))
    return ops

def main():
    t0 = time.time()
    ops: List[UpdateOne] = []
    countries = [c["id"] for c in col_ctry.find({}, {"id":1})]
    for i, iso3 in enumerate(countries, 1):
        ops.extend(build_for_country(iso3))
        # flush periodically
        if len(ops) >= 5000:
            upsert_many(ops); ops = []
        if i % 25 == 0:
            print(f"[persona] countries processed: {i}/{len(countries)}")
    if ops: upsert_many(ops)
    col_ps.create_index([("country", ASCENDING), ("year", ASCENDING)], name="country_year")
    print(f"Done in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
