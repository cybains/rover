# rover-learn/backend/app.py

import asyncio
from datetime import datetime
from typing import List, Optional

from bson import ObjectId
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware

from .db import get_db
from .glossary import load_glossary
from .models import SessionCreate


# ---------- helpers ----------

def to_jsonable(x):
    """Recursively convert Mongo/BSON types (e.g., ObjectId) into JSON-safe values."""
    if isinstance(x, ObjectId):
        return str(x)
    if isinstance(x, list):
        return [to_jsonable(i) for i in x]
    if isinstance(x, dict):
        return {k: to_jsonable(v) for k, v in x.items()}
    return x


# ---------- app & middleware ----------

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
    # Ensure DB connection (and any index creation you do in get_db)
    await get_db()


# ---------- health ----------

@app.get("/health")
async def health():
    return {"service": "backend", "status": "ok"}


# ---------- sessions ----------

@app.post("/sessions/start")
async def start_session(payload: SessionCreate = Body(...)):
    """
    Accepts: { "title": string | null }
    Creates a live session and returns it with string _id and segmentsCount: 0
    """
    db = await get_db()
    now = datetime.utcnow()
    title: Optional[str] = (payload.title or "").strip() if payload and payload.title else ""

    doc = {
        "title": title if title else "Untitled Session",
        "createdAt": now,
        "updatedAt": now,
        "status": "live",
    }
    res = await db.sessions.insert_one(doc)

    out = {
        "_id": str(res.inserted_id),
        "title": doc["title"],
        "createdAt": doc["createdAt"],
        "updatedAt": doc["updatedAt"],
        "status": doc["status"],
        "segmentsCount": 0,
    }
    return to_jsonable(out)


@app.post("/sessions/stop")
async def stop_session(payload: dict = Body(...)):
    session_id = (payload or {}).get("sessionId")
    if not session_id:
        return {"error": "sessionId required"}

    db = await get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        return {"error": "invalid sessionId"}

    await db.sessions.update_one(
        {"_id": oid},
        {"$set": {"status": "stopped", "updatedAt": datetime.utcnow()}},
    )
    return {"status": "stopped", "sessionId": session_id}


@app.get("/sessions")
async def list_sessions():
    """
    Returns sessions sorted by createdAt desc, including a robust segmentsCount that
    matches both string and ObjectId-stored sessionId (in case of historical data).
    """
    db = await get_db()
    sessions: List[dict] = []
    cursor = db.sessions.find().sort("createdAt", -1)

    async for s in cursor:
        sid_str = str(s["_id"])
        # robust count: match segments that stored sessionId as string OR ObjectId
        count = await db.segments.count_documents({"sessionId": {"$in": [sid_str, s["_id"]]}})
        sessions.append(
            {
                "_id": sid_str,
                "title": s.get("title"),
                "createdAt": s.get("createdAt"),
                "updatedAt": s.get("updatedAt"),
                "status": s.get("status"),
                "segmentsCount": count,
            }
        )
    return to_jsonable(sessions)


@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """
    Returns the session (with string _id) and its segments (sorted by idx asc),
    plus segmentsCount.
    """
    db = await get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        return {}

    session = await db.sessions.find_one({"_id": oid})
    if not session:
        return {}

    session["_id"] = str(session["_id"])
    segments: List[dict] = []

    # Prefer segments stored with string sessionId; also accept old ObjectId-stored
    cursor = db.segments.find({"sessionId": {"$in": [session_id, oid]}}).sort("idx", 1)
    async for seg in cursor:
        seg["_id"] = str(seg["_id"])
        segments.append(seg)

    session["segments"] = segments
    session["segmentsCount"] = len(segments)
    return to_jsonable(session)


# ---------- glossary ----------

@app.get("/glossary")
async def glossary():
    return to_jsonable(load_glossary())


# ---------- realtime (mock stream) ----------

@app.websocket("/realtime")
async def realtime(ws: WebSocket):
    """
    Mock realtime stream:
    - Requires ?sessionId=<id>
    - Emits one segment per second with incremental idx
    - Inserts segment into Mongo
    - Bumps session.updatedAt each time
    """
    await ws.accept()
    session_id = ws.query_params.get("sessionId")
    if not session_id:
        await ws.close()
        return

    # Validate ObjectId for session updates; keep string in segments
    try:
        session_oid = ObjectId(session_id)
    except Exception:
        await ws.close()
        return

    db = await get_db()

    # Start idx after existing segments (support both string and ObjectId-stored)
    idx = await db.segments.count_documents({"sessionId": {"$in": [session_id, session_oid]}})

    try:
        while True:
            # ensure session is still live
            session = await db.sessions.find_one({"_id": session_oid})
            if not session or session.get("status") != "live":
                break

            # build mock segment
            segment = {
                "sessionId": session_id,   # keep as STRING for the UI
                "idx": idx,
                "tStart": float(idx),      # simple increasing times for the mock
                "tEnd": float(idx) + 1.0,
                "lang": "de",
                "speaker": "Speaker 1",
                "textSrc": f"Guten Tag {idx}",
                "textEn": f"[MT-MOCK] Guten Tag {idx}",
                "partial": False,
                "confidence": 0.92,
            }

            # insert & attach _id (as string) for completeness
            ins = await db.segments.insert_one(segment)
            segment["_id"] = str(ins.inserted_id)

            # bump updatedAt on the session
            await db.sessions.update_one(
                {"_id": session_oid},
                {"$set": {"updatedAt": datetime.utcnow()}},
            )

            # send JSON-safe payload
            await ws.send_json(to_jsonable(segment))

            idx += 1
            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        # client closed; just exit
        pass
    finally:
        # make sure the socket is closed
        try:
            await ws.close()
        except Exception:
            pass
