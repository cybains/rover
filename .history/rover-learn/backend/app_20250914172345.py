import asyncio
from datetime import datetime
from typing import List

from bson import ObjectId
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .db import get_db
from .glossary import load_glossary
from .models import SessionCreate
from bson import ObjectId
from fastapi.encoders import jsonable_encoder

def to_jsonable(x):
    # handles dicts/lists/ObjectId recursively
    if isinstance(x, ObjectId):
        return str(x)
    if isinstance(x, list):
        return [to_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    return x

app = FastAPI()

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    # Ensure DB connection and indexes
    await get_db()


@app.get("/health")
async def health():
    return {"service": "backend", "status": "ok"}


@app.post("/sessions/start")
async def start_session(payload: SessionCreate):
    db = await get_db()
    now = datetime.utcnow()
    data = {
        "title": payload.title or "Untitled Session",
        "createdAt": now,
        "updatedAt": now,
        "status": "live",
    }
    res = await db.sessions.insert_one(data)
    data["_id"] = str(res.inserted_id)
    data["segmentsCount"] = 0
    return data


@app.post("/sessions/stop")
async def stop_session(payload: dict):
    session_id = payload.get("sessionId")
    if not session_id:
        return {"error": "sessionId required"}
    db = await get_db()
    updated = datetime.utcnow()
    await db.sessions.update_one(
        {"_id": ObjectId(session_id)},
        {"$set": {"status": "stopped", "updatedAt": updated}},
    )
    return {"status": "stopped"}


@app.get("/sessions")
async def list_sessions():
    db = await get_db()
    sessions: List[dict] = []
    cursor = db.sessions.find().sort("createdAt", -1)
    async for s in cursor:
        count = await db.segments.count_documents({
    "sessionId": {"$in": [str(s["_id"]), s["_id"]]}
})
        sessions.append(
            {
                "_id": str(s["_id"]),
                "title": s.get("title"),
                "createdAt": s.get("createdAt"),
                "updatedAt": s.get("updatedAt"),
                "status": s.get("status"),
                "segmentsCount": count,
            }
        )
    return sessions


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    db = await get_db()
    session = await db.sessions.find_one({"_id": ObjectId(session_id)})
    if not session:
        return {}
    session["_id"] = str(session["_id"])
    segments: List[dict] = []
    cursor = db.segments.find({"sessionId": session_id}).sort("idx", 1)
    async for seg in cursor:
        seg["_id"] = str(seg["_id"])
        segments.append(seg)
    session["segments"] = segments
    session["segmentsCount"] = len(segments)
    return session


@app.get("/glossary")
async def glossary():
    return load_glossary()


@app.websocket("/realtime")
async def realtime(ws: WebSocket):
    await ws.accept()
    session_id = ws.query_params.get("sessionId")
    if not session_id:
        await ws.close()
        return
    db = await get_db()
    # determine starting index
    idx = await db.segments.count_documents({"sessionId": session_id})
    try:
        while True:
            session = await db.sessions.find_one({"_id": ObjectId(session_id)})
            if not session or session.get("status") != "live":
                break
            segment = {
                "sessionId": session_id,
                "idx": idx,
                "tStart": 0.0,
                "tEnd": 1.2,
                "lang": "de",
                "speaker": "Speaker 1",
                "textSrc": f"Guten Tag {idx}",
                "textEn": f"[MT-MOCK] Guten Tag {idx}",
                "partial": False,
                "confidence": 0.92,
            }
            await db.segments.insert_one(segment)
            await db.sessions.update_one(
                {"_id": ObjectId(session_id)},
                {"$set": {"updatedAt": datetime.utcnow()}},
            )
            await ws.send_json(segment)
            idx += 1
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
    finally:
        await ws.close()
