# services/ingest/connectors/worldbank_ingest.py
import os
import requests
from pymongo import MongoClient
from dotenv import load_dotenv
from services.ingest.connectors.worldbank import WorldBank

# --- Load .env ---
load_dotenv()
MONGO_URI = os.getenv("MONGO_URI")

if not MONGO_URI:
    raise ValueError("⚠️ MONGO_URI not found in .env file")

# --- MongoDB Setup ---
client = MongoClient(MONGO_URI)
db = client["worldbank"]
collection = db["countries"]

# --- World Bank Setup ---
wb = WorldBank()

def get_all_indicators():
    """Fetch all indicator IDs from World Bank."""
    url = "https://api.worldbank.org/v2/indicator?format=json&per_page=20000"
    r = requests.get(url)
    r.raise_for_status()
    data = r.json()
    return [ind["id"] for ind in data[1]]  # list of indicator codes

def ingest_country(country_code: str):
    indicators = get_all_indicators()
    print(f"Found {len(indicators)} indicators")

    for ind in indicators:
        try:
            payload = wb.fetch_indicator(indicator=ind, country=country_code, date="1960:2024")
            docs = wb.normalize(payload, indicator=ind)

            if docs:
                collection.insert_many([d.__dict__ for d in docs], ordered=False)
                print(f"Inserted {len(docs)} docs for {country_code} - {ind}")
        except Exception as e:
            print(f"Error fetching {ind} for {country_code}: {e}")

if __name__ == "__main__":
    for country in ["AT", "PT"]:  # Austria, Portugal
        ingest_country(country)
