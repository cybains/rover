import os
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

_client: Optional[AsyncIOMotorClient] = None
_db: Optional[AsyncIOMotorDatabase] = None


async def get_db() -> AsyncIOMotorDatabase:
    global _client, _db
    if _db is None:
        uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
        _client = AsyncIOMotorClient(uri)
        db_name = os.getenv("MONGODB_DB", "rover")
        _db = _client[db_name]
        # ensure indexes
        await _db.segments.create_index([("sessionId", 1), ("idx", 1)], unique=True)
    return _db
