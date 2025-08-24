# services/ingest/connectors/evidence_bundle_materializer.py
import os, time
from typing import Optional
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne, ASCENDING

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI: raise SystemExit("MONGO_URI missing")
cli = MongoClient(MONGO_URI)
db = cli["worldbank_core"]
views = db["country_views"]
ps    = db["persona_scores"]
bund  = db["evidence_bundles"]

def find_ps_for_year(country: str, y: int) -> Optional[dict]:
    doc = ps.find_one({"country": country, "year": y}, {"_id":0})
    if doc: return doc
    # fallback to latest <= y
    cur = ps.find({"country": country, "year": {"$lte": y}}, {"_id":0}).sort("year",-1).limit(1)
    return next(cur, None)

def run():
    ops = []
    t0 = time.time()
    for v in views.find({}, {"_id":0}):
        country = v["country"]["id"]
        # pick bundle year as the max headline year available
        yrs = [h.get("year") for h in v.get("headlines", []) if isinstance(h.get("year"), int)]
        if not yrs: continue
        y = max(yrs)
        psdoc = find_ps_for_year(country, y)
        doc = {
            "_id": f"{country}|{y}",
            "country": v["country"],
            "year": y,
            "headlines": v["headlines"],
            "personas": psdoc["personas"] if psdoc else None,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "notes": {
                "persona_year_used": psdoc["year"] if psdoc else None,
            }
        }
        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": doc}, upsert=True))
        if len(ops) >= 500:
            bund.bulk_write(ops, ordered=False); ops = []
    if ops: bund.bulk_write(ops, ordered=False)
    bund.create_index([("country.id", ASCENDING), ("year", ASCENDING)], name="country_year")
    print("evidence_bundles ready in", round(time.time()-t0,1), "s")

if __name__ == "__main__":
    run()
