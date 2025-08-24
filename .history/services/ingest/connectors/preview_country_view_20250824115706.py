# services/ingest/connectors/preview_country_view.py
import os, pprint
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
cli = MongoClient(os.getenv("MONGO_URI"))
doc = cli["worldbank_core"]["country_views"].find_one({"_id":"AUT"})
pprint.pp({"country": doc["country"], "first_5_headlines": doc["headlines"][:5]})
