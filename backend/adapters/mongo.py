import os
from datetime import datetime
from typing import List, Dict, Any

from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

MONGO_URI = os.getenv("MONGO_URI", "mongodb://127.0.0.1:27017/rover")
client = AsyncIOMotorClient(MONGO_URI)
db = client.get_default_database()
sessions_col = db["sessions"]
segments_col = db["segments"]

async def init_indexes():
    await segments_col.create_index([("sessionId", 1), ("idx", 1)])

def _normalize_id(doc):
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

async def create_session(title: str) -> Dict[str, Any]:
    doc = {"title": title, "createdAt": datetime.utcnow(), "status": "active"}
    res = await sessions_col.insert_one(doc)
    doc["_id"] = str(res.inserted_id)
    return doc

async def set_session_status(session_id: str, status: str):
    await sessions_col.update_one({"_id": ObjectId(session_id)}, {"$set": {"status": status}})

async def list_sessions() -> List[Dict[str, Any]]:
    cur = sessions_col.find().sort("createdAt", -1)
    return [_normalize_id(d) async for d in cur]

async def get_last_active_session():
    doc = await sessions_col.find_one({"status": "active"}, sort=[("createdAt", -1)])
    return _normalize_id(doc)

async def append_segments(session_id: str, segs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segs:
        return []
    last = await segments_col.find({"sessionId": session_id}).sort("idx", -1).limit(1).to_list(1)
    start = last[0]["idx"] + 1 if last else 0
    for i, s in enumerate(segs):
        s["sessionId"] = session_id
        s["idx"] = start + i
    await segments_col.insert_many(segs)
    return segs

async def list_segments(session_id: str) -> List[Dict[str, Any]]:
    cur = segments_col.find({"sessionId": session_id}).sort("idx", 1)
    items = []
    async for d in cur:
        items.append(_normalize_id(d))
    return items
