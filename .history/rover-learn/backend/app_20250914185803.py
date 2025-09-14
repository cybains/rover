# rover-learn/backend/app.py

from __future__ import annotations

import json
import math
import sys
import asyncio
import subprocess
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import requests
from bson import ObjectId
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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

MT_URL = "http://localhost:4002"
SESSION_SOCKETS: dict[str, set[WebSocket]] = defaultdict(set)
CAPTURE_PROC: dict[str, subprocess.Popen] = {}

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


# ---------- capture ----------

async def _stop_capture(session_id: str):
    proc = CAPTURE_PROC.pop(session_id, None)
    if not proc:
        return

    def _terminate(p: subprocess.Popen):
        try:
            p.terminate()
            p.wait(timeout=3)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass

    await asyncio.to_thread(_terminate, proc)


@app.post("/capture/start")
async def start_capture(payload: dict = Body(...)):
    session_id = (payload or {}).get("sessionId")
    source = (payload or {}).get("source", "auto")
    if not session_id:
        return {"error": "sessionId required"}

    # already running?
    proc = CAPTURE_PROC.get(session_id)
    if proc and proc.poll() is None:
        raise HTTPException(status_code=409, detail="capture already running")

    repo_root = Path(__file__).resolve().parents[1]  # rover-learn/
    agent = repo_root / "services" / "capture" / "agent.py"
    cmd = [
        sys.executable,
        str(agent),
        "--session",
        session_id,
        "--asr",
        "http://localhost:4001",
        "--api",
        "http://localhost:4000",
        "--source",
        source,
    ]
    # Use repo root as CWD so relative imports/paths inside the agent work
    proc = subprocess.Popen(cmd, cwd=str(repo_root))
    CAPTURE_PROC[session_id] = proc
    return {"ok": True, "pid": proc.pid}


@app.post("/capture/stop")
async def stop_capture(payload: dict = Body(...)):
    session_id = (payload or {}).get("sessionId")
    if not session_id:
        return {"error": "sessionId required"}
    await _stop_capture(session_id)
    return {"ok": True}


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
    await _stop_capture(session_id)
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
        raise HTTPException(status_code=404)

    session = await db.sessions.find_one({"_id": oid})
    if not session:
        raise HTTPException(status_code=404)

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
        Path(__file__).resolve().parents[1]
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
    base = Path(__file__).resolve().parents[1] / "exports" / "sessions"
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


# ---------- ingestion ----------

@app.post("/ingest_segment")
async def ingest_segment(payload: dict = Body(...)):
    """
    Ingest a single segment:
      - normalize fields (lang, timestamps, text)
      - translate DE->EN via Marian (liberal match: 'de'/'deu'/etc.)
      - persist, update session.updatedAt, and broadcast over WS
    """
    session_id = (payload or {}).get("sessionId")
    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId required")

    db = await get_db()
    try:
        session_oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid sessionId")

    # --- robust normalization ---
    lang_raw = (payload.get("lang") or "").strip().lower()

    def _safe_time(x, default: float) -> float:
        try:
            xf = float(x)
            if math.isnan(xf) or math.isinf(xf):
                return default
            return xf
        except Exception:
            return default

    t_start = _safe_time(payload.get("tStart"), 0.0)
    t_end = _safe_time(payload.get("tEnd"), max(t_start, t_start + 1.0))  # at least +1s

    text_src = str(payload.get("textSrc", "") or "").strip()
    speaker = str(payload.get("speaker", "") or "Speaker 1")
    partial = bool(payload.get("partial", False))
    try:
        confidence = float(payload.get("confidence", 0.9))
    except Exception:
        confidence = 0.9

    segment = {
        "sessionId": session_id,
        "idx": int(payload.get("idx", 0)),
        "tStart": t_start,
        "tEnd": t_end,
        "lang": lang_raw,
        "speaker": speaker,
        "textSrc": text_src,
        "partial": partial,
        "confidence": confidence,
        # textEn filled below
    }

    # --- translation ---
    try:
        if lang_raw.startswith("de"):  # handles "de", "deu", "de-DE"
            r = requests.post(
                f"{MT_URL}/translate",
                json={"text": segment["textSrc"], "src_lang": "de", "tgt_lang": "en"},
                timeout=10,
            )
            r.raise_for_status()
            segment["textEn"] = r.json().get("translation", segment["textSrc"])
        else:
            segment["textEn"] = f"[MT-MOCK] {segment['textSrc']}"
    except Exception:
        # On any MT error, fall back to echo so ingestion never fails
        segment["textEn"] = segment["textSrc"]

    ins = await db.segments.insert_one(segment)
    segment["_id"] = str(ins.inserted_id)

    await db.sessions.update_one(
        {"_id": session_oid},
        {"$set": {"updatedAt": datetime.utcnow()}},
    )

    # --- broadcast over WS ---
    sockets = SESSION_SOCKETS.get(session_id)
    if sockets:
        dead = set()
        for ws in list(sockets):
            try:
                await ws.send_json(to_jsonable(segment))
            except Exception:
                dead.add(ws)
        for ws in dead:
            sockets.discard(ws)

    return {"ok": True}


# ---------- realtime ----------

@app.websocket("/realtime")
async def realtime(ws: WebSocket):
    await ws.accept()
    session_id = ws.query_params.get("sessionId")
    if not session_id:
        await ws.close()
        return

    SESSION_SOCKETS[session_id].add(ws)
    try:
        # Keep connection open; we only push server->client
        while True:
            await asyncio.sleep(60)
    except WebSocketDisconnect:
        pass
    finally:
        SESSION_SOCKETS[session_id].discard(ws)
        try:
            await ws.close()
        except Exception:
            pass
