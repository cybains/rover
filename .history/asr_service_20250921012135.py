# asr_service.py
"""
ASR microservice (Windows-friendly) for rover-learn.
- GPU (CUDA) faster-whisper
- 0.75s hop, resample to 16k
- Mic / System / Auto (WASAPI loopback)
- SSE events: {chunk_index, start_sec, end_sec, text, final}
- /session/start -> {"session_id": "..."}  (UI expects snake_case)
- /session/stop  <- {"session_id": "..."}
"""

from __future__ import annotations

import asyncio
import atexit
import json
import logging
import os
import queue as queue_module
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Literal

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

# =========================
# Config
# =========================
TARGET_SR = 16000
CHUNK_SEC = 0.75               # ~snappy updates
MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"
BEAM_SIZE = 5
USE_VAD = False                # avoid extra onnxruntime dep
WORD_TS = False                # segment timing only
CAPTURE_DTYPE = "int16"

MIC_HINTS = ["mikrofon", "microphone", "array"]   # case-insensitive
SPK_HINTS = ["speaker", "lautsprecher"]
PREFERRED_HOST = "Windows WASAPI"

logger = logging.getLogger("asr_service")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")

# =========================
# State
# =========================
@dataclass
class SessionState:
    session_id: str
    source: Literal["mic", "system", "auto"] = "auto"
    idx: int = 0
    queue: "queue_module.Queue[Optional[dict]]" = field(default_factory=queue_module.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    finished_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    total_target_samples: int = 0
    active: bool = True

SESSIONS: Dict[str, SessionState] = {}
SESSIONS_LOCK = threading.Lock()

MODEL_LOCK = threading.Lock()
MODEL: Optional[WhisperModel] = None

# =========================
# API models
# =========================
class StopSessionRequest(BaseModel):
    session_id: str

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
# Audio device helpers
# =========================
def _pick_device_for_mic() -> Tuple[Optional[int], str, int, int]:
    """Pick a WASAPI mic if possible, else any WASAPI input, else default input."""
    try:
        hostapis = sd.query_hostapis()
        ha_names = {i: ha.get("name", "") for i, ha in enumerate(hostapis)}
    except Exception as e:
        logger.warning("Unable to query hostapis: %s", e)
        ha_names = {}
    devices = sd.query_devices()
    default_in, _ = sd.default.device

    # WASAPI mic-like by name
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
    """Pick WASAPI playback device for loopback (system audio)."""
    try:
        hostapis = sd.query_hostapis()
        ha_names = {i: ha.get("name", "") for i, ha in enumerate(hostapis)}
    except Exception as e:
        logger.warning("Unable to query hostapis: %s", e)
        ha_names = {}
    devices = sd.query_devices()
    _, default_out = sd.default.device

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

    # fallback default output
    try:
        info = sd.query_devices(default_out)
        if info.get("max_output_channels", 0) > 0:
            host = ha_names.get(info.get("hostapi", -1), "")
            sr = int(round(info.get("default_samplerate") or 48000))
            return default_out, f"[default output] {info['name']} ({host})", 2, sr
    except Exception:
        pass
    return None, "[none]", 0, 48000

# =========================
# DSP helpers
# =========================
def _downmix_to_mono_int16(x: np.ndarray) -> np.ndarray:
    """Accept (N,) or (N,C) int16 -> return (N,) int16 mono."""
    if x.ndim == 1:
        return x
    mono = x.astype(np.int32).mean(axis=1)
    mono = np.clip(mono, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return mono.astype(np.int16)

def _resample_linear(x_f32: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """x_f32 is float32 mono; linear interpolation to dst_rate; returns float32."""
    if x_f32.size == 0 or src_rate == dst_rate:
        return x_f32.astype(np.float32, copy=False)
    n_in = x_f32.shape[0]
    n_out = max(1, int(round(n_in * dst_rate / src_rate)))
    xp = np.linspace(0.0, n_in, num=n_in, endpoint=False, dtype=np.float32)
    x_new = np.linspace(0.0, n_in, num=n_out, endpoint=False, dtype=np.float32)
    y = np.interp(x_new, xp, x_f32.astype(np.float32, copy=False))
    return y.astype(np.float32, copy=False)

# =========================
# Transcribe helpers
# =========================
def _emit_event_ui(session: SessionState, text: str,
                   t_start: float, t_end: float, is_final: bool) -> dict:
    """Emit events in the exact shape the UI expects."""
    ev = {
        "chunk_index": session.idx,
        "start_sec": round(t_start, 3),
        "end_sec": round(t_end, 3),
        "text": (text or "").strip(),
        "final": bool(is_final),
    }
    session.queue.put(ev)
    # IMPORTANT: bump index only when final is sent
    if is_final:
        session.idx += 1
    return ev

def _transcribe(audio_16k: np.ndarray) -> tuple[str, float, float]:
    """Return (text, rel_start, rel_end) for a single chunk at 16 kHz float32."""
    with MODEL_LOCK:
        if MODEL is None:
            _init_model()
        model = MODEL
    assert model is not None

    # light silence gate
    if audio_16k.size == 0:
        return "", 0.0, 0.0
    rms = float(np.sqrt(np.mean(np.square(audio_16k))) + 1e-12)
    if rms < 1e-7:
        dur = audio_16k.size / TARGET_SR
        return "", 0.0, dur

    segments, info = model.transcribe(
        audio=audio_16k,       # mono float32 16 kHz
        language=None,
        beam_size=BEAM_SIZE,
        vad_filter=USE_VAD,
        word_timestamps=WORD_TS,
    )

    text_parts = []
    rel_s = 0.0
    rel_e = audio_16k.size / TARGET_SR
    got_s = False

    for seg in segments:
        if seg.text:
            text_parts.append(seg.text.strip())
        if (seg.start is not None) and not got_s:
            rel_s = float(seg.start); got_s = True
        if seg.end is not None:
            rel_e = float(seg.end)

    return " ".join(text_parts).strip(), rel_s, rel_e

# =========================
# Capture thread
# =========================
def _capture_worker(session: SessionState) -> None:
    use_loopback = False
    if session.source == "mic":
        dev_idx, label, in_ch, dev_sr = _pick_device_for_mic()
    elif session.source == "system":
        dev_idx, label, _, dev_sr = _pick_device_for_system()
        in_ch = 2; use_loopback = True
    else:
        dev_idx, label, in_ch, dev_sr = _pick_device_for_mic()
        if dev_idx is None:
            dev_idx, label, _, dev_sr = _pick_device_for_system()
            in_ch = 2; use_loopback = True

    if dev_idx is None or (in_ch <= 0 and not use_loopback):
        logger.error("No usable audio device for source=%s", session.source)
        t = session.total_target_samples / TARGET_SR
        _emit_event_ui(session, "", t, t, True)
        session.queue.put(None); session.active = False; session.finished_event.set()
        return

    dev_sr = int(max(8000, min(192000, round(dev_sr))))
    in_channels = 2 if use_loopback else min(in_ch, 2)
    logger.info("Using device #%s %s @ %d Hz (source=%s)", dev_idx, label, dev_sr, session.source)

    frames_per_read = int(dev_sr * 0.5)                 # ~0.5s IO reads
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
            logger.info("Session %s started capture (GPU %s %s)", session.session_id, DEVICE, COMPUTE_TYPE)
            while not session.stop_event.is_set():
                frames, overflowed = stream.read(frames_per_read)
                if overflowed:
                    logger.warning("Audio overflow (session %s)", session.session_id)
                buf.extend(frames.tobytes())

                while len(buf) >= chunk_samples_dev * bytes_per_frame:
                    raw = bytes(buf[: chunk_samples_dev * bytes_per_frame])
                    del buf[: chunk_samples_dev * bytes_per_frame]

                    arr = np.frombuffer(raw, dtype=np.int16)
                    if in_channels > 1:
                        arr = arr.reshape(-1, in_channels)
                    mono_i16 = _downmix_to_mono_int16(arr)
                    mono_f32 = mono_i16.astype(np.float32) / 32768.0
                    mono_16k = _resample_linear(mono_f32, dev_sr, TARGET_SR)

                    t0 = session.total_target_samples / TARGET_SR
                    session.total_target_samples += mono_16k.size

                    text, rs, re = _transcribe(mono_16k)

                    # Emit partial then final with SAME chunk_index
                    _emit_event_ui(session, text, t0 + rs, t0 + re, is_final=False)
                    _emit_event_ui(session, text, t0 + rs, t0 + re, is_final=True)

            # flush leftover
            leftover = (len(buf) // bytes_per_frame) * bytes_per_frame
            if leftover > 0:
                raw = bytes(buf[:leftover]); del buf[:leftover]
                arr = np.frombuffer(raw, dtype=np.int16)
                if in_channels > 1:
                    arr = arr.reshape(-1, in_channels)
                mono_i16 = _downmix_to_mono_int16(arr)
                mono_f32 = mono_i16.astype(np.float32) / 32768.0
                mono_16k = _resample_linear(mono_f32, dev_sr, TARGET_SR)
            else:
                mono_16k = np.array([], dtype=np.float32)

            t0 = session.total_target_samples / TARGET_SR
            session.total_target_samples += mono_16k.size
            if mono_16k.size:
                text, rs, re = _transcribe(mono_16k)
                _emit_event_ui(session, text, t0 + rs, t0 + re, is_final=False)
                _emit_event_ui(session, text, t0 + rs, t0 + re, is_final=True)
            else:
                _emit_event_ui(session, "", t0, t0, is_final=True)

        logger.info("Session %s finished capture", session.session_id)
    except Exception as e:
        logger.exception("Session %s error: %s", session.session_id, e)
        t = session.total_target_samples / TARGET_SR
        _emit_event_ui(session, "", t, t, True)
    finally:
        session.queue.put(None)
        session.active = False
        session.finished_event.set()

# =========================
# Lifecycle helpers
# =========================
def _ensure_single_active() -> None:
    with SESSIONS_LOCK:
        act = [s for s in SESSIONS.values() if s.active and not s.stop_event.is_set()]
        if act:
            raise HTTPException(status_code=409, detail="ASR session already running")

def start_session(source: Literal["mic", "system", "auto"] = "auto") -> SessionState:
    _ensure_single_active()
    session_id = str(uuid.uuid4())
    ss = SessionState(session_id=session_id, source=source)
    with SESSIONS_LOCK:
        SESSIONS[session_id] = ss
    t = threading.Thread(target=_capture_worker, args=(ss,), name=f"asr-{session_id}", daemon=True)
    ss.thread = t
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
# FastAPI app
# =========================
app = FastAPI(title="ASR Service", version="2.1.0")

# CORS for your Next.js dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # adjust if your UI uses a different origin
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

@app.post("/session/start")
async def api_start(request: Request):
    """
    Starts a new capture/transcription session.
    Optional query param: ?source=mic|system|auto   (default: auto)
    Returns: {"session_id": "..."}
    """
    qsrc = request.query_params.get("source", "auto").lower()
    source: Literal["mic", "system", "auto"] = "auto"
    if qsrc in ("mic", "system", "auto"):
        source = qsrc  # type: ignore
    ss = start_session(source=source)
    return JSONResponse({"session_id": ss.session_id, "source": ss.source})

@app.post("/session/stop")
async def api_stop(body: StopSessionRequest):
    stop_session(body.session_id)
    return {"status": "stopped", "session_id": body.session_id}

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
