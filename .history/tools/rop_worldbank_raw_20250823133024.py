import os
from pymongo import MongoClient
from dotenv import load_dotenv
load_dotenv()
c = MongoClient(os.getenv("MONGO_URI"))
c.drop_database("worldbank_raw")
print("Dropped worldbank_raw")
