# services/ingest/connectors/preview_evidence_aut.py
import os, pprint
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
cli = MongoClient(os.getenv("MONGO_URI"))
col = cli["worldbank_core"]["evidence_bundles"]
doc = col.find({"country.id":"AUT"}).sort("year",-1).limit(1)[0]
pprint.pp({
  "country": doc["country"],
  "year": doc["year"],
  "kpis": [{ "code": h["code"], "y": h["year"], "v": h["value"], "p": h["pctl"]["world"] } for h in doc["headlines"][:6]],
  "personas": {k: round(v["score"],1) if v["score"] is not None else None for k,v in doc["personas"].items()},
  "persona_year_used": doc["notes"]["persona_year_used"],
})
