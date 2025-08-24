# backend/app/main.py
import os
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI missing")

client = MongoClient(MONGO_URI)
db = client["worldbank_core"]
col_countries = db["countries"]
col_bundles   = db["evidence_bundles"]

app = FastAPI(title="Rover API", version="0.1.0")

# CORS (adjust for your frontend origin)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_bundle(iso3: str, year: Optional[int] = None):
    iso3 = iso3.upper()
    if year is not None:
        doc = col_bundles.find_one({"_id": f"{iso3}|{year}"}, {"_id": 0})
        if doc:
            return doc
        # fallback: latest <= year
        cur = col_bundles.find({"country.id": iso3, "year": {"$lte": year}}, {"_id": 0}) \
                         .sort("year", -1).limit(1)
        doc = next(cur, None)
        if doc:
            doc.setdefault("notes", {})["fallback_from_year_query"] = year
            return doc
        raise HTTPException(404, detail="No bundle for that country/year")
    # latest overall
    cur = col_bundles.find({"country.id": iso3}, {"_id": 0}).sort("year", -1).limit(1)
    doc = next(cur, None)
    if not doc:
        raise HTTPException(404, detail="No bundle for that country")
    return doc

@app.get("/api/countries")
def countries():
    cur = col_countries.find({}, {"_id":0, "id":1, "name":1, "region":1, "incomeLevel":1, "iso2Code":1})
    return [{"id": d["id"], "name": d.get("name"), "region": d.get("region"),
             "income": d.get("incomeLevel"), "iso2": d.get("iso2Code")} for d in cur]

@app.get("/api/country/{iso3}")
def country_latest(iso3: str):
    return get_bundle(iso3)

@app.get("/api/country/{iso3}/{year}")
def country_year(iso3: str, year: int):
    return get_bundle(iso3, year)
