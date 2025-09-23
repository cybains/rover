# asr_service.py
"""
ASR microservice (Windows-friendly) for rover-learn.
- GPU (CUDA) faster-whisper
- 0.75s chunks, resample to 16k
- Mic / System / Auto (WASAPI + loopback)
- SSE events compatible with backend ingest: type/final, tStart/tEnd, textSrc, lang, idx
- Backward-compatible /session/start: accepts JSON body OR query params.
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel, Field

# =========================
# Config
# =========================
TARGET_SR = 16000
CHUNK_SEC = 0.75              # hop (~snappy UX)
MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
DEVICE = "cuda"               # GPU only
COMPUTE_TYPE = "int8_float16"
BEAM_SIZE = 5
USE_VAD = False               # keep off (no onnxruntime)
WORD_TS = False               # we only need segment text/timing
CAPTURE_DTYPE = "int16"       # 16-bit PCM
MIC_HINTS = ["mikrofon", "microphone", "array"]     # case-insensitive
SPK_HINTS = ["speaker", "lautsprecher"]             # for system loopback
PREFERRED_HOST = "Windows WASAPI"

# logging
logger = logging.getLogger("asr")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

# =========================
# State
# =========================
@dataclass
class SessionState:
    session_id: str
    idx: int = 0
    queue: "queue.Queue[Optional[dict]]" = field(default_factory=queue.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    finished_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    total_target_samples: int = 0
    active: bool = True
    source: Literal["mic", "system", "auto"] = "auto"

SESSIONS: Dict[str, SessionState] = {}
SESSIONS_LOCK = threading.Lock()

MODEL_LOCK = threading.Lock()
MODEL: Optional[WhisperModel] = None

# =========================
# Models (API)
# =========================
class StartBody(BaseModel):
    sessionId: Optional[str] = Field(None, description="Use the app/backend sessionId")
    source: Optional[Literal["mic", "system", "auto"]] = "auto"

class StopBody(BaseModel):
    sessionId: str

# =========================
# Model init / teardown
# =========================
def _init_model() -> None:
    global MODEL
    with MODEL_LOCK:
        if MODEL is not None:
            return
        MODEL = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
        logger.info("Loaded model '%s' on %s (%s)", MODEL_NAME, DEVICE, COMPUTE_TYPE)

def _close_model() -> None:
    global MODEL
    with MODEL_LOCK:
        MODEL = None

atexit.register(_close_model)

# =========================
# Audio helpers
# =========================
def _pick_device_for_mic() -> Tuple[Optional[int], str, int, int]:
    """Prefer WASAPI input matching mic-like names; else any WASAPI input; else default input."""
    try:
        hostapis = sd.query_hostapis()
        ha_names = {i: ha.get("name", "") for i, ha in enumerate(hostapis)}
    except Exception as e:
        logger.warning("Unable to query hostapis: %s", e)
        ha_names = {}

    devices = sd.query_devices()
    default_in, _ = sd.default.device
    # mic-like on WASAPI
    for i, info in enumerate(devices):
        if info.get("max_input_channels", 0) <= 0:
            continue
        host = ha_names.get(info.get("hostapi", -1), "")
        if PREFERRED_HOST.lower() in host.lower():
            name_l = str(info.get("name", "")).lower()
            if any(k in name_l for k in MIC_HINTS):
                sr = int(round(info.get("default_samplerate") or 48000))
                return i, f"[WASAPI mic] {info['name']}", int(info["max_input_channels"]), sr
    # any WASAPI input
    for i, info in enumerate(devices):
        if info.get("max_input_channels", 0) <= 0:
            continue
        host = ha_names.get(info.get("hostapi", -1), "")
        if PREFERRED_HOST.lower() in host.lower():
            sr = int(round(info.get("default_samplerate") or 48000))
            return i, f"[WASAPI mic-any] {info['name']}", int(info["max_input_channels"]), sr
    # default input
    try:
        info = sd.query_devices(default_in)
        if info.get("max_input_channels", 0) > 0:
            sr = int(round(info.get("default_samplerate") or 48000))
            host = ha_names.get(info.get("hostapi", -1), "")
            return default_in, f"[default mic] {info['name']} ({host})", int(info["max_input_channels"]), sr
    except Exception:
        pass
    return None, "[none]", 0, 48000

def _pick_device_for_system() -> Tuple[Optional[int], str, int, int]:
    """Pick a WASAPI playback device for loopback (system audio)."""
    try:
        hostapis = sd.query_hostapis()
        ha_names = {i: ha.get("name", "") for i, ha in enumerate(hostapis)}
    except Exception as e:
        logger.warning("Unable to query hostapis: %s", e)
        ha_names = {}

    devices = sd.query_devices()
    _, default_out = sd.default.device
    # WASAPI speakers-like output
    candidates = []
    for i, info in enumerate(devices):
        if info.get("max_output_channels", 0) <= 0:
            continue
        host = ha_names.get(info.get("hostapi", -1), "")
        if PREFERRED_HOST.lower() in host.lower():
            name_l = str(info.get("name", "")).lower()
            score = 0
            if any(k in name_l for k in SPK_HINTS):
                score += 2
            if "realtek" in name_l:
                score += 1
            candidates.append((score, i, info, host))
    if candidates:
        candidates.sort(reverse=True)
        _, i, info, host = candidates[0]
        sr = int(round(info.get("default_samplerate") or 48000))
        return i, f"[WASAPI loopback] {info['name']} ({host})", 2, sr
    # fallback: default output
    try:
        info = sd.query_devices(default_out)
        if info.get("max_output_channels", 0) > 0:
            host = ha_names.get(info.get("hostapi", -1), "")
            sr = int(round(info.get("default_samplerate") or 48000))
            return default_out, f"[default output] {info['name']} ({host})", 2, sr
    except Exception:
        pass
    return None, "[none]", 0, 48000

def _downmix_to_mono_int16(x: np.ndarray) -> np.ndarray:
    if x.ndim == 1:
        return x
    mono = x.astype(np.int32).mean(axis=1)
    mono = np.clip(mono, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return mono.astype(np.int16)

def _resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if x.size == 0 or src_rate == dst_rate:
        return x.astype(np.float32, copy=False)
    n_in = x.shape[0]
    n_out = max(1, int(round(n_in * dst_rate / src_rate)))
    xp = np.linspace(0.0, n_in, num=n_in, endpoint=False)
    x_new = np.linspace(0.0, n_in, num=n_out, endpoint=False)
    y = np.interp(x_new, xp, x.astype(np.float32))
    return y.astype(np.float32, copy=False)

# =========================
# Transcribe
# =========================
def _emit_event(
    session: SessionState,
    text: str,
    lang: Optional[str],
    t_start: float,
    t_end: float,
    is_final: bool,
) -> dict:
    ev = {
        "type": "final" if is_final else "final",  # finals only for now
        "sessionId": session.session_id,
        "idx": session.idx,
        "tStart": round(t_start, 3),
        "tEnd": round(t_end, 3),
        "lang": (lang or "").lower() if lang else "",
        "textSrc": text.strip(),
    }
    session.queue.put(ev)
    session.idx += 1
    return ev

def _transcribe(
    audio_16k: np.ndarray,
) -> Tuple[str, Optional[str], float, float]:
    """Return (text, lang, rel_start, rel_end) in seconds relative to this chunk."""
    with MODEL_LOCK:
        if MODEL is None:
            _init_model()
        model = MODEL
    assert model is not None

    text_parts = []
    rel_start = 0.0
    rel_end = audio_16k.size / TARGET_SR
    detected_lang: Optional[str] = None

    # tiny energy gate to avoid pure silence
    rms = float(np.sqrt(np.mean(np.square(audio_16k))) + 1e-12)
    if rms < 1e-7:
        return "", None, rel_start, rel_end

    segments, info = model.transcribe(
        audio=audio_16k,
        language=None,          # auto-detect
        beam_size=BEAM_SIZE,
        vad_filter=USE_VAD,
        word_timestamps=WORD_TS,
    )
    try:
        if info and getattr(info, "language", None):
            detected_lang = str(info.language).lower()
    except Exception:
        detected_lang = None

    got_ts = False
    for seg in segments:
        if seg.text:
            text_parts.append(seg.text.strip())
        if (seg.start is not None) and not got_ts:
            rel_start = float(seg.start)
            got_ts = True
        if seg.end is not None:
            rel_end = float(seg.end)

    return " ".join(text_parts).strip(), detected_lang, rel_start, rel_end

def _capture_worker(session: SessionState) -> None:
    """
    Capture → chunk → resample → transcribe → enqueue SSE events.
    Supports mic/system/auto (WASAPI loopback for system).
    """
    # Choose device & loopback flag
    use_loopback = False
    if session.source == "mic":
        dev_idx, label, in_ch, dev_sr = _pick_device_for_mic()
    elif session.source == "system":
        dev_idx, label, out_ch, dev_sr = _pick_device_for_system()
        in_ch = 2
        use_loopback = True
    else:  # auto
        dev_idx, label, in_ch, dev_sr = _pick_device_for_mic()
        if dev_idx is None:
            dev_idx, label, out_ch, dev_sr = _pick_device_for_system()
            in_ch = 2
            use_loopback = True

    if dev_idx is None or (in_ch <= 0 and not use_loopback):
        logger.error("No usable audio device found for source=%s", session.source)
        _emit_event(session, "", None, 0.0, 0.0, True)
        session.queue.put(None)
        session.active = False
        session.finished_event.set()
        return

    dev_sr = int(max(8000, min(192000, round(dev_sr))))
    in_channels = 2 if use_loopback else min(in_ch, 2)
    logger.info("Using %s #%s @ %d Hz (source=%s)", label, dev_idx, dev_sr, session.source)

    # chunking
    frames_per_read = int(dev_sr * 0.5)                     # 0.5s read cadence
    chunk_samples_dev = int(dev_sr * CHUNK_SEC)
    bytes_per_frame = in_channels * np.dtype(CAPTURE_DTYPE).itemsize
    buf = bytearray()

    extra = None
    if use_loopback:
        try:
            extra = sd.WasapiSettings(loopback=True)
        except Exception:
            extra = None

    try:
        with sd.InputStream(
            device=dev_idx,
            samplerate=dev_sr,
            channels=in_channels if not use_loopback else 2,
            dtype=CAPTURE_DTYPE,
            blocksize=0,
            latency="low",
            extra_settings=extra,
        ) as stream:
            logger.info("Session %s started capture (%s/%s)", session.session_id, DEVICE, COMPUTE_TYPE)
            while not session.stop_event.is_set():
                frames, overflow = stream.read(frames_per_read)
                if overflow:
                    logger.warning("Audio overflow (session %s)", session.session_id)
                buf.extend(frames.tobytes())

                # process full chunks
                while len(buf) >= chunk_samples_dev * bytes_per_frame:
                    raw = bytes(buf[: chunk_samples_dev * bytes_per_frame])
                    del buf[: chunk_samples_dev * bytes_per_frame]

                    arr = np.frombuffer(raw, dtype=np.int16)
                    if in_channels > 1:
                        arr = arr.reshape(-1, in_channels)
                    mono_i16 = _downmix_to_mono_int16(arr)
                    mono_16k = _resample_linear(mono_i16, dev_sr, TARGET_SR)

                    # time mapping: running sample count at 16k
                    t0 = session.total_target_samples / TARGET_SR
                    session.total_target_samples += mono_16k.size

                    text, lang, rel_s, rel_e = _transcribe(mono_16k)
                    _emit_event(session, text, lang, t0 + rel_s, t0 + rel_e, False)

            # flush leftover as final
            leftover_frames = (len(buf) // bytes_per_frame) * bytes_per_frame
            if leftover_frames > 0:
                raw = bytes(buf[:leftover_frames])
                del buf[:leftover_frames]
                arr = np.frombuffer(raw, dtype=np.int16)
                if in_channels > 1:
                    arr = arr.reshape(-1, in_channels)
                mono_i16 = _downmix_to_mono_int16(arr)
                mono_16k = _resample_linear(mono_i16, dev_sr, TARGET_SR)
            else:
                mono_16k = np.array([], dtype=np.float32)

            t0 = session.total_target_samples / TARGET_SR
            session.total_target_samples += mono_16k.size
            text, lang, rel_s, rel_e = _transcribe(mono_16k) if mono_16k.size else ("", None, 0.0, 0.0)
            _emit_event(session, text, lang, t0 + rel_s, t0 + rel_e, True)

        logger.info("Session %s finished capture", session.session_id)
    except Exception as e:
        logger.exception("Session %s error: %s", session.session_id, e)
        t = session.total_target_samples / TARGET_SR
        _emit_event(session, "", None, t, t, True)
    finally:
        session.queue.put(None)
        session.active = False
        session.finished_event.set()

# =========================
# Session lifecycle
# =========================
def _ensure_single_active() -> None:
    with SESSIONS_LOCK:
        act = [s for s in SESSIONS.values() if s.active and not s.stop_event.is_set()]
        if act:
            raise HTTPException(status_code=409, detail="ASR session already running")

def start_session(session_id: str, source: Literal["mic", "system", "auto"]) -> SessionState:
    _ensure_single_active()
    ss = SessionState(session_id=session_id, source=source)
    with SESSIONS_LOCK:
        SESSIONS[session_id] = ss
    t = threading.Thread(target=_capture_worker, args=(ss,), name=f"asr-{session_id}", daemon=True)
    ss.thread = t
    t.start
    t.start()
    return ss

def stop_session(session_id: str) -> None:
    with SESSIONS_LOCK:
        ss = SESSIONS.get(session_id)
    if not ss:
        raise HTTPException(status_code=404, detail="Session not found")
    logger.info("Stopping session %s", session_id)
    ss.stop_event.set()
    if ss.thread and ss.thread.is_alive():
        ss.thread.join(timeout=5)
    ss.finished_event.wait(timeout=5)
    with SESSIONS_LOCK:
        SESSIONS.pop(session_id, None)
    logger.info("Session %s stopped & cleaned", session_id)

# =========================
# FastAPI
# =========================
app = FastAPI(title="ASR Service", version="2.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def _startup():
    _init_model()

@app.on_event("shutdown")
async def _shutdown():
    with SESSIONS_LOCK:
        running = list(SESSIONS.values())
    for s in running:
        try:
            s.stop_event.set()
            if s.thread and s.thread.is_alive():
                s.thread.join(timeout=5)
            s.finished_event.wait(timeout=5)
        except Exception:
            pass
    _close_model()

@app.get("/health")
async def health():
    return {"status": "ok"}

def _extract_start_params(body: Optional[StartBody], request: Request) -> Tuple[str, Literal["mic","system","auto"]]:
    """Accept JSON body OR query params for backward compatibility."""
    session_id = None
    source: Literal["mic","system","auto"] = "auto"
    # from body
    if body and body.sessionId:
        session_id = body.sessionId.strip()
    if body and body.source:
        source = body.source
    # fallback: query params
    qs_sid = request.query_params.get("sessionId")
    if not session_id and qs_sid:
        session_id = qs_sid.strip()
    qs_src = request.query_params.get("source")
    if qs_src in ("mic", "system", "auto"):
        source = qs_src  # allow overriding via query
    # final check
    if not session_id:
        raise HTTPException(status_code=400, detail="sessionId required (send JSON {sessionId,...} or use ?sessionId=...)")
    return session_id, source

@app.post("/session/start")
async def api_start(request: Request, body: Optional[StartBody] = None):
    session_id, source = _extract_start_params(body, request)
    ss = start_session(session_id, source)  # raises if already active
    return JSONResponse({"sessionId": ss.session_id, "source": ss.source})

@app.post("/session/stop")
async def api_stop(body: StopBody):
    stop_session(body.sessionId)
    return {"status": "stopped", "sessionId": body.sessionId}

@app.get("/session/{session_id}/stream")
async def api_stream(session_id: str, request: Request):
    with SESSIONS_LOCK:
        ss = SESSIONS.get(session_id)
    if not ss:
        raise HTTPException(status_code=404, detail="Session not found")
    q = ss.queue

    async def gen():
        loop = asyncio.get_running_loop()
        while True:
            if await request.is_disconnected():
                logger.info("Client disconnected (session %s)", session_id)
                break
            item = await loop.run_in_executor(None, q.get)
            if item is None:
                break
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")
