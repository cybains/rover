import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient

_client: Optional[AsyncIOMotorClient] = None
_db = None


async def get_db():
    global _client, _db
    if _db is not None:
        return _db

    uri = (
        os.getenv("MONGODB_URI")
        or os.getenv("MONGO_URI")
        or "mongodb://localhost:27017"
    )
    dbname = (
        os.getenv("MONGODB_DB")
        or os.getenv("MONGO_DB")
        or os.getenv("JOBS_DB_NAME")
        or "lab"
    )

    _client = AsyncIOMotorClient(uri)
    _db = _client[dbname]

    seg = _db.segments
    info = await seg.index_information()
    name = "sessionId_1_idx_1"
    wants_unique = True
    if name in info and bool(info[name].get("unique")) != wants_unique:
        await seg.drop_index(name)
    await seg.create_index([("sessionId", 1), ("idx", 1)], name=name, unique=True)

    documents = _db.documents
    await documents.create_index([("uploadedAt", -1)])
    await documents.create_index([("linkedSessions", 1)])

    trash = _db.trash
    await trash.create_index([("resourceType", 1)])
    await trash.create_index([("resourceId", 1)])
    await trash.create_index("expiresAt", expireAfterSeconds=0)

    return _db
