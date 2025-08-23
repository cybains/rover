import os, pprint
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
c = MongoClient(os.getenv("MONGO_URI")); db = c["worldbank_raw"]
obs = db["observations"]

print("Total observations:", obs.estimated_document_count())
print("\nTop countries by rows:")
for x in obs.aggregate([{"$group":{"_id":"$country.id","n":{"$sum":1}}},{"$sort":{"n":-1}}, {"$limit":10}]): print(x)
print("\nTop indicators by rows:")
for x in obs.aggregate([{"$group":{"_id":"$indicator.id","n":{"$sum":1}}},{"$sort":{"n":-1}}, {"$limit":10}]): print(x)
print("\nSample:")
pprint.pprint(obs.find_one({}, {"_id":0, "indicator":1, "country":1, "date":1, "value":1}))
