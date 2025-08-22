# services/ingest/connectors/preview_docs.py
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import pprint

# Load .env
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("⚠️ MONGO_URI not found in .env file")

# MongoDB setup
client = MongoClient(MONGO_URI)
db = client["worldbank"]
collection = db["countries"]

# Fetch a few docs
docs = collection.find().limit(3)

print("=== Sample Documents ===")
for doc in docs:
    pprint.pprint(doc)
    print("\n---\n")
