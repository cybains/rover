# rover-learn/backend/db.py
import os
from motor.motor_asyncio import AsyncIOMotorClient

_client = None
_db = None

async def get_db():
    global _client, _db
    if _db is not None:               # <-- explicit check
        return _db

    uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
    dbname = os.getenv("MONGODB_DB", "rover")

    _client = AsyncIOMotorClient(uri)
    _db = _client[dbname]

    # Ensure indexes (drop conflicting one if unique flag differs)
    seg = _db.segments
    info = await seg.index_information()
    name = "sessionId_1_idx_1"
    wants_unique = True
    if name in info and bool(info[name].get("unique")) != wants_unique:
        await seg.drop_index(name)
    await seg.create_index([("sessionId", 1), ("idx", 1)], name=name, unique=True)

    return _db
