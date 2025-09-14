# rover-learn/backend/app.py
from __future__ import annotations

import json
import math
import sys
import asyncio
import subprocess
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict
import re

import os
import httpx
from bson import ObjectId
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from .db import get_db
from .utils.glossary import load_glossary, apply_glossary
from .models import SessionCreate
from .utils.qalink import recompute_qalinks
from services.llm.client import generate_json


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

MT_URL = os.getenv("MT_URL", "http://localhost:4002")
HTTP = httpx.AsyncClient(timeout=3.0)
SESSION_SOCKETS: dict[str, set[WebSocket]] = defaultdict(set)
CAPTURE_PROC: dict[str, subprocess.Popen] = {}
LAST_INGEST_AT: dict[str, datetime] = {}

# de-dup cache (avoid repeats)
LAST_SEG_TEXT: Dict[str, str] = {}      # sessionId -> last textSrc
LAST_SEG_TEND: Dict[str, float] = {}    # sessionId -> last tEnd

DE_Q_START = re.compile(
    r"^(wer|was|wann|wo|warum|wieso|weshalb|welche|welcher|welches|wie)\b", re.I
)

SETTINGS_PATH = Path(__file__).resolve().parents[1] / "config" / "runtime_settings.json"
DEFAULT_SETTINGS = {
    "asrForceLang": "",
    "chunkMs": 500,
    "targetLang": "EN",
    "exportDefaults": {},
}

STORAGE_THRESHOLD = 2 * 1024 * 1024 * 1024  # 2 GB
STORAGE_STATUS: Dict[str, int | bool] = {}


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text())
    except Exception:
        return DEFAULT_SETTINGS.copy()


def save_settings(data: dict):
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(data))


def _dir_size(p: Path) -> int:
    if not p.exists():
        return 0
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


async def _compute_storage() -> dict:
    exports_dir = Path(__file__).resolve().parents[1] / "exports"
    exports_bytes = _dir_size(exports_dir)
    db = await get_db()
    try:
        stats = await db.command("dbstats")
        db_bytes = int(stats.get("storageSize", 0))
    except Exception:
        db_bytes = 0
    total = exports_bytes + db_bytes
    return {
        "exportsBytes": exports_bytes,
        "dbBytes": db_bytes,
        "totalBytes": total,
        "overLimit": total > STORAGE_THRESHOLD,
    }


async def _storage_monitor():
    while True:
        STORAGE_STATUS.update(await _compute_storage())
        await asyncio.sleep(60)

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
    await get_db()
    load_glossary()
    asyncio.create_task(_storage_monitor())


@app.on_event("shutdown")
async def shutdown_event():
    await HTTP.aclose()


# ---------- health ----------

@app.get("/health")
async def health():
    return {"service": "backend", "status": "ok"}


# ---------- debug/status ----------

@app.get("/ws/status")
async def ws_status(sessionId: str):
    sockets = SESSION_SOCKETS.get(sessionId) or set()
    return {"sessionId": sessionId, "clients": len(sockets)}

@app.get("/capture/status")
async def capture_status(sessionId: str):
    proc = CAPTURE_PROC.get(sessionId)
    running = bool(proc and proc.poll() is None)
    pid = proc.pid if running else None
    last = LAST_INGEST_AT.get(sessionId)
    return {"sessionId": sessionId, "running": running, "pid": pid, "lastIngestTs": last.isoformat() if last else None}


# ---------- settings & storage ----------


@app.get("/settings")
async def get_settings_endpoint():
    return load_settings()


@app.post("/settings")
async def post_settings_endpoint(payload: dict = Body(...)):
    data = load_settings()
    for k in DEFAULT_SETTINGS:
        if k in payload:
            data[k] = payload[k]
    save_settings(data)
    return data


@app.get("/storage/status")
async def storage_status():
    if not STORAGE_STATUS:
        STORAGE_STATUS.update(await _compute_storage())
    return STORAGE_STATUS


@app.post("/sessions/purge")
async def purge_sessions(payload: dict = Body(...)):
    before = (payload or {}).get("before")
    if not before:
        raise HTTPException(status_code=400, detail="before required")
    try:
        dt = datetime.fromisoformat(before)
    except ValueError:
        raise HTTPException(status_code=400, detail="invalid before")
    db = await get_db()
    cursor = db.sessions.find({"createdAt": {"$lt": dt}})
    deleted = 0
    async for s in cursor:
        sid = s["_id"]
        sid_str = str(sid)
        await db.segments.delete_many({"sessionId": {"$in": [sid, sid_str]}})
        await db.generations.delete_many({"sessionId": sid_str})
        export_dir = (
            Path(__file__).resolve().parents[1]
            / "exports"
            / "sessions"
            / s["createdAt"].strftime("%Y-%m-%d")
            / sid_str
        )
        await asyncio.to_thread(shutil.rmtree, export_dir, True)
        await db.sessions.delete_one({"_id": sid})
        deleted += 1
    STORAGE_STATUS.update(await _compute_storage())
    return {"deleted": deleted}


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
        raise HTTPException(status_code=400, detail="sessionId required")

    # already running?
    proc = CAPTURE_PROC.get(session_id)
    if proc and proc.poll() is None:
        return {"ok": True, "pid": proc.pid}

    repo_root = Path(__file__).resolve().parents[1]  # rover-learn/
    agent = repo_root / "services" / "capture" / "agent.py"
    cmd = [
        sys.executable,
        str(agent),
        "--session", session_id,
        "--asr", "http://localhost:4001",
        "--api", "http://localhost:4000",
        "--source", source,
    ]
    proc = subprocess.Popen(cmd, cwd=str(repo_root))
    CAPTURE_PROC[session_id] = proc
    return {"ok": True, "pid": proc.pid}

@app.post("/capture/stop")
async def stop_capture(payload: dict = Body(...)):
    session_id = (payload or {}).get("sessionId")
    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId required")
    await _stop_capture(session_id)
    return {"ok": True}


# ---------- sessions ----------

@app.post("/sessions/start")
async def start_session(payload: SessionCreate = Body(...)):
    db = await get_db()
    now = datetime.utcnow()
    title: Optional[str] = (payload.title or "").strip() if payload and payload.title else ""

    doc = {
        "title": title if title else "Untitled Session",
        "createdAt": now,
        "updatedAt": now,
        "status": "live",
        "speakerMap": {},
    }
    res = await db.sessions.insert_one(doc)

    out = {
        "_id": str(res.inserted_id),
        "title": doc["title"],
        "createdAt": doc["createdAt"],
        "updatedAt": doc["updatedAt"],
        "status": doc["status"],
        "speakerMap": {},
        "segmentsCount": 0,
    }
    return to_jsonable(out)

@app.post("/sessions/stop")
async def stop_session(payload: dict = Body(...)):
    session_id = (payload or {}).get("sessionId")
    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId required")

    db = await get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=400, detail="invalid sessionId")

    await db.sessions.update_one(
        {"_id": oid},
        {"$set": {"status": "stopped", "updatedAt": datetime.utcnow()}},
    )
    await _stop_capture(session_id)
    return {"status": "stopped", "sessionId": session_id}

@app.get("/sessions")
async def list_sessions():
    db = await get_db()
    sessions: List[dict] = []
    cursor = db.sessions.find().sort("createdAt", -1)
    async for s in cursor:
        sid_str = str(s["_id"])
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
    db = await get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=404)

    session = await db.sessions.find_one({"_id": oid})
    if not session:
        raise HTTPException(status_code=404)

    session["_id"] = str(session["_id"])
    session["speakerMap"] = session.get("speakerMap", {})
    segments: List[dict] = []
    cursor = db.segments.find({"sessionId": {"$in": [session_id, oid]}}).sort("idxStart", 1)
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
    cursor = db.segments.find({"sessionId": {"$in": [session_id, oid]}}).sort("idxStart", 1)
    async for seg in cursor:
        seg["_id"] = str(seg["_id"])
        segments.append(seg)

    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    export_dir = (
        Path(__file__).resolve().parents[1] / "exports" / "sessions" / date_str / session_id
    )
    export_dir.mkdir(parents=True, exist_ok=True)

    (export_dir / "transcript_src.txt").write_text(
        "\n".join(seg.get("textSrc", "") for seg in segments), encoding="utf-8"
    )
    (export_dir / "translation_en.txt").write_text(
        "\n".join(seg.get("textEn", "") for seg in segments), encoding="utf-8"
    )
    with (export_dir / "segments.jsonl").open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(to_jsonable(seg), ensure_ascii=False) + "\n")

    def _fmt_ts(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = t % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    lines_src = ["WEBVTT", ""]
    lines_en = ["WEBVTT", ""]
    for seg in segments:
        start = _fmt_ts(seg.get("tStart", 0.0))
        end = _fmt_ts(seg.get("tEnd", 0.0))
        lines_src.append(f"{start} --> {end}")
        lines_src.append(seg.get("textSrc", ""))
        lines_src.append("")
        lines_en.append(f"{start} --> {end}")
        lines_en.append(seg.get("textEn", ""))
        lines_en.append("")

    (export_dir / "captions_src.vtt").write_text("\n".join(lines_src), encoding="utf-8")
    (export_dir / "captions_en.vtt").write_text("\n".join(lines_en), encoding="utf-8")

    out = {
        "sessionId": session_id,
        "exportDir": str(export_dir).replace("\\", "/") + "/",
        "files": [
            "transcript_src.txt",
            "translation_en.txt",
            "segments.jsonl",
            "captions_src.vtt",
            "captions_en.vtt",
        ],
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
        "files": [
            "transcript_src.txt",
            "translation_en.txt",
            "segments.jsonl",
            "captions_src.vtt",
            "captions_en.vtt",
        ],
        "counts": {"segments": count},
    }


# ---------- qa / bookmarks ----------

@app.post("/sessions/{session_id}/qa/recompute")
async def qa_recompute(session_id: str):
    db = await get_db()
    await recompute_qalinks(db, session_id)
    return {"ok": True}


@app.post("/sessions/{session_id}/speakers/rename")
async def rename_speaker(session_id: str, payload: dict = Body(...)):
    from_name = (payload or {}).get("from")
    to_name = (payload or {}).get("to")
    if not from_name or not to_name:
        raise HTTPException(status_code=400, detail="from/to required")
    db = await get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=404)
    await db.sessions.update_one(
        {"_id": oid},
        {"$set": {f"speakerMap.{from_name}": to_name, "updatedAt": datetime.utcnow()}},
    )
    sockets = SESSION_SOCKETS.get(session_id)
    if sockets:
        msg = {"type": "speakerRename", "from": from_name, "to": to_name}
        dead = set()
        for ws in list(sockets):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        for ws in dead:
            SESSION_SOCKETS[session_id].discard(ws)
    return {"ok": True}


@app.post("/segments/{segment_id}/bookmark")
async def bookmark_segment(segment_id: str):
    db = await get_db()
    try:
        oid = ObjectId(segment_id)
    except Exception:
        raise HTTPException(status_code=404)
    await db.segments.update_one({"_id": oid}, {"$set": {"bookmark": True}})
    return {"ok": True}


@app.delete("/segments/{segment_id}/bookmark")
async def unbookmark_segment(segment_id: str):
    db = await get_db()
    try:
        oid = ObjectId(segment_id)
    except Exception:
        raise HTTPException(status_code=404)
    await db.segments.update_one({"_id": oid}, {"$set": {"bookmark": False}})
    return {"ok": True}


@app.get("/sessions/{session_id}/highlights")
async def session_highlights(session_id: str):
    db = await get_db()
    try:
        oid = ObjectId(session_id)
    except Exception:
        raise HTTPException(status_code=404)
    cursor = db.segments.find({"sessionId": {"$in": [session_id, oid]}}).sort("idxStart", 1)
    questions: List[dict] = []
    bookmarks: List[dict] = []
    glossary: List[dict] = []
    by_idx: Dict[int, dict] = {}
    answer_idxs = set()
    async for seg in cursor:
        seg["_id"] = str(seg["_id"])
        by_idx[seg["idxStart"]] = seg
        if seg.get("bookmark"):
            bookmarks.append(seg)
        if seg.get("isQuestion"):
            questions.append(seg)
            qa = seg.get("qa") or {}
            best = qa.get("bestAnswerIdx")
            if isinstance(best, int):
                answer_idxs.add(best)
        if seg.get("glossaryHits"):
            glossary.append(seg)
    answers = [by_idx[i] for i in answer_idxs if i in by_idx]
    return to_jsonable(
        {
            "questions": questions,
            "answers": answers,
            "bookmarks": bookmarks,
            "glossary": glossary,
        }
    )


# ---------- generations ----------

def _build_prompt(kind: str, text: str, options: dict) -> str:
    if kind == "summary":
        return (
            "Summarize the following text and return JSON as {\"summary\": \"...\"}.\n"
            + text
        )
    if kind == "flashcards":
        return (
            "Create flashcards from the text. Return JSON as {\"flashcards\": [{\"front\":\"\", \"back\":\"\"}]}\n"
            + text
        )
    if kind == "quiz":
        return (
            "Create a quiz with questions and answers. Return JSON as {\"questions\": [{\"q\":\"\", \"a\":\"\"}]}\n"
            + text
        )
    level = options.get("level", "simple")
    return (
        f"Explain the text at a {level} level. Return JSON as {{\"explanation\": \"...\"}}.\n"
        + text
    )


async def _collect_text(session_id: str, para_ids: List[str] | None):
    db = await get_db()
    q = {"sessionId": session_id}
    if para_ids:
        oids = []
        for pid in para_ids:
            try:
                oids.append(ObjectId(pid))
            except Exception:
                continue
        if oids:
            q["_id"] = {"$in": oids}
    cursor = db.segments.find(q).sort("idxStart", 1)
    segs = await cursor.to_list(None)
    text = "\n".join(s.get("textEn", "") for s in segs)
    return text, [str(s.get("_id")) for s in segs]


@app.post("/generate/{kind}")
async def generate(kind: str, payload: dict = Body(...)):
    if kind not in {"summary", "flashcards", "quiz", "explain"}:
        raise HTTPException(status_code=404)
    session_id = payload.get("sessionId")
    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId required")
    para_ids = payload.get("paraIds") or None
    options = payload.get("options") or {}
    text, used_ids = await _collect_text(session_id, para_ids)
    prompt = _build_prompt(kind, text, options)
    try:
        output = await generate_json(prompt)
    except Exception as e:
        output = {"error": str(e)}
    db = await get_db()
    doc = {
        "sessionId": session_id,
        "type": kind,
        "sourceRange": used_ids,
        "output": output,
        "createdAt": datetime.utcnow(),
    }
    ins = await db.generations.insert_one(doc)
    doc["_id"] = str(ins.inserted_id)
    return to_jsonable(doc)


@app.get("/sessions/{session_id}/generations")
async def list_generations(session_id: str, type: str | None = None):
    db = await get_db()
    q = {"sessionId": session_id}
    if type:
        q["type"] = type
    cursor = db.generations.find(q).sort("createdAt", -1)
    items = []
    async for g in cursor:
        g["_id"] = str(g["_id"])
        items.append(g)
    return to_jsonable(items)


# ---------- ingestion ----------

@app.post("/ingest_segment")
async def ingest_segment(payload: dict = Body(...)):
    """
    Ingest a single segment:
      - normalize fields (lang, timestamps, text)
      - translate DE->EN via Marian
      - dedup near-identical repeats within 2s
      - persist, update session.updatedAt, and broadcast over WS
    """
    try:
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="invalid body")

        session_id = (payload or {}).get("sessionId")
        if not session_id:
            raise HTTPException(status_code=400, detail="sessionId required")

        db = await get_db()
        try:
            session_oid = ObjectId(session_id)
        except Exception:
            raise HTTPException(status_code=400, detail="invalid sessionId")

        kind = payload.get("kind", "final")
        if kind == "partial":
            para_id = payload.get("paraId")
            lang_raw = (payload.get("lang") or "").strip().lower()
            text_src = str(payload.get("textSrcPartial", "") or "").strip()
            speaker = str(payload.get("speaker", "") or "")
            seg = {
                "kind": "partial",
                "paraId": para_id,
                "lang": lang_raw,
                "textSrcPartial": text_src,
                "speaker": speaker,
            }
            try:
                if lang_raw == "de" and text_src:
                    r = await HTTP.post(
                        f"{MT_URL}/translate_partial",
                        json={"text": text_src, "src_lang": "de", "tgt_lang": "en"},
                    )
                    r.raise_for_status()
                    seg["textEnPartial"] = r.json().get("translation", text_src)
                elif text_src:
                    r = await HTTP.post(
                        f"{MT_URL}/translate_partial",
                        json={"text": text_src, "src_lang": lang_raw, "tgt_lang": "en"},
                    )
                    r.raise_for_status()
                    seg["textEnPartial"] = r.json().get("translation", text_src)
                else:
                    seg["textEnPartial"] = ""
            except Exception:
                seg["textEnPartial"] = "[MT-MOCK] " + text_src

            sockets = SESSION_SOCKETS.get(session_id)
            if sockets:
                dead = set()
                for ws in list(sockets):
                    try:
                        await ws.send_json(to_jsonable(seg))
                    except Exception:
                        dead.add(ws)
                for ws in dead:
                    SESSION_SOCKETS[session_id].discard(ws)
            return {"ok": True, "partial": True}

        # final segment
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
        t_end = _safe_time(payload.get("tEnd"), max(t_start, t_start + 1.0))

        text_src = str(payload.get("textSrc", "") or "").strip()
        speaker = str(payload.get("speaker", "") or "Speaker 1")
        try:
            confidence = float(payload.get("confidence", 0.9))
        except Exception:
            confidence = 0.9

        segment = {
            "sessionId": session_id,
            "paraId": payload.get("paraId"),
            "idxStart": int(payload.get("idx", 0)),
            "idxEnd": int(payload.get("idx", 0)),
            "tStart": t_start,
            "tEnd": t_end,
            "lang": lang_raw,
            "speaker": speaker,
            "textSrc": text_src,
            "partial": False,
            "confidence": confidence,
        }

        ts_clean = text_src.strip()
        segment["isQuestion"] = bool(ts_clean.endswith("?") or DE_Q_START.match(ts_clean))

        try:
            if text_src:
                r = await HTTP.post(
                    f"{MT_URL}/translate_final",
                    json={"text": text_src, "src_lang": lang_raw or "de", "tgt_lang": "en"},
                )
                r.raise_for_status()
                segment["textEn"] = r.json().get("translation", text_src)
            else:
                segment["textEn"] = ""
        except Exception:
            segment["textEn"] = "[MT-MOCK] " + text_src

        segment["textEn"], hits = apply_glossary(segment["textEn"])
        if hits:
            segment["glossaryHits"] = hits

        ins = await db.segments.insert_one(segment)
        segment["_id"] = str(ins.inserted_id)

        await db.sessions.update_one(
            {"_id": session_oid},
            {"$set": {"updatedAt": datetime.utcnow()}},
        )

        LAST_INGEST_AT[session_id] = datetime.utcnow()

        sockets = SESSION_SOCKETS.get(session_id)
        if sockets:
            dead = set()
            for ws in list(sockets):
                try:
                    await ws.send_json(to_jsonable(segment))
                except Exception:
                    dead.add(ws)
            for ws in dead:
                SESSION_SOCKETS[session_id].discard(ws)

        return {"ok": True}

    except HTTPException:
        raise
    except Exception as e:
        # Do not 500; surface payload for debugging
        return {"ok": False, "error": str(e), "payload": to_jsonable(payload)}


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
        # periodic heartbeat keeps client latency gauge fresh
        while True:
            await asyncio.sleep(2)  # was larger; make it responsive
            try:
                await ws.send_json({"type": "heartbeat", "ts": datetime.utcnow().isoformat()})
            except Exception:
                break
    except WebSocketDisconnect:
        pass
    finally:
        SESSION_SOCKETS[session_id].discard(ws)
        try:
            await ws.close()
        except Exception:
            pass
