import os, time
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
c = MongoClient(os.getenv("MONGO_URI")); db = c["worldbank_raw"]
state, inds, obs, src = db["ingest_state"], db["indicators"], db["observations"], db["sources"]

sources = [s["id"] for s in src.find({}, {"id":1})]
total = sum(inds.count_documents({"source_id": sid}) for sid in sources)
while True:
    done = state.count_documents({"status":"done"})
    rows = obs.estimated_document_count()
    pct  = (done/total*100) if total else 100
    print(f"Indicators done: {done}/{total} ({pct:.1f}%) | Observations: ~{rows:,}")
    time.sleep(10)
