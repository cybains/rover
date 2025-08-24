# services/ingest/connectors/preview_persona_aut.py
import os, pprint
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
cli = MongoClient(os.getenv("MONGO_URI"))
col = cli["worldbank_core"]["persona_scores"]

doc = col.find({"country":"AUT"}).sort("year",-1).limit(1)[0]
pprint.pp({
  "country": doc["country"],
  "year": doc["year"],
  "job_seeker": doc["personas"]["job_seeker"]["score"],
  "entrepreneur": doc["personas"]["entrepreneur"]["score"],
  "digital_nomad": doc["personas"]["digital_nomad"]["score"],
  "expat_family": doc["personas"]["expat_family"]["score"],
  "coverage": {k: v["coverage"] for k,v in doc["personas"].items()},
})
