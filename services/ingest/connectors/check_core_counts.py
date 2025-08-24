# services/ingest/connectors/check_core_counts.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
client = MongoClient(os.getenv("MONGO_URI"))
col = client["worldbank_core"]["observations"]

pipeline = [
  {"$group": {"_id": "$indicator.id", "n": {"$sum": 1}}},
  {"$sort": {"_id": 1}}
]
for d in col.aggregate(pipeline):
    print(f"{d['_id']}: {d['n']}")
