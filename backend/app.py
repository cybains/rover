import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List

from backend.utils.config import get_config
from backend.adapters import mongo
from backend.utils import exports

cfg = get_config()
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[f"http://localhost:{cfg['web_port']}"],
    allow_methods=["*"],
    allow_headers=["*"],
)

rooms: Dict[str, List[WebSocket]] = {}

@app.on_event("startup")
async def startup():
    await mongo.init_indexes()

async def _notify(session_id: str, segs: List[dict]):
    for ws in rooms.get(session_id, []):
        await ws.send_json(segs)

@app.websocket("/realtime")
async def realtime(ws: WebSocket):
    session_id = ws.query_params.get("sessionId")
    await ws.accept()
    rooms.setdefault(session_id, []).append(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        rooms[session_id].remove(ws)

@app.post("/sessions/start")
async def start_session(payload: dict):
    title = payload.get("title", "Session")
    return await mongo.create_session(title)

@app.post("/sessions/stop")
async def stop_session(payload: dict):
    sid = payload["id"]
    await mongo.set_session_status(sid, "stopped")
    segs = await mongo.list_segments(sid)
    export_dir = cfg["export_dir"]
    exports.write_txts(sid, segs, export_dir)
    exports.write_jsonl(sid, segs, export_dir)
    return {"status": "stopped"}

@app.get("/sessions")
async def sessions():
    return await mongo.list_sessions()

@app.get("/sessions/{sid}/segments")
async def get_segments(sid: str):
    return await mongo.list_segments(sid)

@app.post("/segments")
async def post_segments(payload: dict):
    sid = payload["sessionId"]
    segs = payload.get("segments", [])
    saved = await mongo.append_segments(sid, segs)
    await _notify(sid, saved)
    return {"count": len(saved)}

@app.post("/mock/stream")
async def mock_stream(payload: dict):
    sid = payload["sessionId"]

    async def emit():
        samples = [
            {"textSrc": "Hallo", "textEn": "Hello", "tStart": 0, "tEnd": 1, "lang": "de", "speaker": "A", "partial": False},
            {"textSrc": "Wie geht's?", "textEn": "How are you?", "tStart": 1, "tEnd": 2, "lang": "de", "speaker": "A", "partial": False},
        ]
        for s in samples:
            await _notify(sid, [s])
            await asyncio.sleep(0.5)

    asyncio.create_task(emit())
    return {"status": "streaming"}
