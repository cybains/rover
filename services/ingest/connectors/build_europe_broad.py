# services/ingest/connectors/build_europe_broad.py
import os, json
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise SystemExit("MONGO_URI missing")

client = MongoClient(MONGO_URI)
col = client["worldbank_core"]["countries"]

EU27 = {
    "AUT","BEL","BGR","HRV","CYP","CZE","DNK","EST","FIN","FRA","DEU","GRC",
    "HUN","IRL","ITA","LVA","LTU","LUX","MLT","NLD","POL","PRT","ROU","SVK","SVN","ESP","SWE"
}
EFTA = {"ISL","LIE","NOR","CHE"}
UK   = {"GBR"}
WBALK= {"ALB","BIH","MNE","MKD","SRB","XKX"}  # Kosovo = XKX in WB
E_NEI= {"UKR","MDA","BLR","RUS","TUR"}
CAUC = {"ARM","AZE","GEO"}
MICRO= {"AND","MCO","SMR"}  # keep if present in DB

CURATED = EU27 | EFTA | UK | WBALK | E_NEI | CAUC | MICRO

# Intersect with what's actually in your DB
present = {d["id"] for d in col.find({}, {"id":1})}
final_set = sorted(CURATED & present)

out = []
name_by_id = {d["id"]: d.get("name","") for d in col.find({"id":{"$in":final_set}}, {"id":1,"name":1})}
for iso3 in final_set:
    out.append({"id": iso3, "name": name_by_id.get(iso3, iso3)})

os.makedirs("config", exist_ok=True)
with open("config/europe_broad.json","w",encoding="utf-8") as f:
    json.dump({"countries": out, "count": len(out)}, f, ensure_ascii=False, indent=2)

print(f"Wrote config/europe_broad.json with {len(out)} countries.")
