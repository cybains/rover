# rover-learn/backend/app.py

import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from bson import ObjectId
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests

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

MT_URL = "http://localhost:4002"
ASR_URL = "http://localhost:4001"

# simple mixed-language samples for the mock stream
SAMPLE_TEXT = [
    "Hallo und guten Morgen",
    "This is an English sentence",
    "Der schnelle Fuchs springt",
    "Another English line",
]

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


# ---------- exports ----------

@app.post("/export/{session_id}")
async def export_session(session_id: str):
    db = await get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=404)

    session = await db.sessions.find_one({"_id": oid})
    if not session:
        raise HTTPException(status_code=404)

    segments: List[dict] = []
    cursor = db.segments.find({"sessionId": {"$in": [session_id, oid]}}).sort("idx", 1)
    async for seg in cursor:
        seg["_id"] = str(seg["_id"])
        segments.append(seg)

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    export_dir = (
        Path(__file__).resolve().parent.parent
        / "exports"
        / "sessions"
        / date_str
        / session_id
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    (export_dir / "transcript_src.txt").write_text(
        "\n".join(seg.get("textSrc", "") for seg in segments),
        encoding="utf-8",
    )
    (export_dir / "translation_en.txt").write_text(
        "\n".join(seg.get("textEn", "") for seg in segments),
        encoding="utf-8",
    )
    with (export_dir / "segments.jsonl").open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(to_jsonable(seg), ensure_ascii=False) + "\n")

    out = {
        "sessionId": session_id,
        "exportDir": str(export_dir).replace("\\", "/") + "/",
        "files": ["transcript_src.txt", "translation_en.txt", "segments.jsonl"],
        "counts": {"segments": len(segments)},
    }
    return out


@app.get("/exports/{session_id}")
async def get_export(session_id: str):
    base = Path(__file__).resolve().parent.parent / "exports" / "sessions"
    if not base.exists():
        raise HTTPException(status_code=404)

    export_dir = None
    for date_dir in base.iterdir():
        candidate = date_dir / session_id
        if candidate.exists():
            export_dir = candidate
            break
    if not export_dir:
        raise HTTPException(status_code=404)

    seg_file = export_dir / "segments.jsonl"
    count = 0
    if seg_file.exists():
        with seg_file.open(encoding="utf-8") as f:
            for _ in f:
                count += 1

    return {
        "sessionId": session_id,
        "exportDir": str(export_dir).replace("\\", "/") + "/",
        "files": ["transcript_src.txt", "translation_en.txt", "segments.jsonl"],
        "counts": {"segments": count},
    }

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

            sample = SAMPLE_TEXT[idx % len(SAMPLE_TEXT)]
            try:
                r = requests.post(
                    f"{ASR_URL}/transcribe",
                    json={"text": sample, "idx": idx},
                    timeout=10,
                )
                seg = r.json()
            except Exception:
                seg = {
                    "idx": idx,
                    "tStart": float(idx),
                    "tEnd": float(idx) + 0.9,
                    "lang": "en",
                    "speaker": "Speaker 1",
                    "textSrc": sample,
                    "partial": False,
                    "confidence": 0.0,
                }

            segment = {**seg, "sessionId": session_id, "textEn": seg["textSrc"]}

            if segment["lang"] == "de":
                try:
                    r = requests.post(
                        f"{MT_URL}/translate",
                        json={"text": segment["textSrc"], "src_lang": "de", "tgt_lang": "en"},
                        timeout=10,
                    )
                    segment["textEn"] = r.json().get("translation", segment["textSrc"])
                except Exception:
                    segment["textEn"] = segment["textSrc"]

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
