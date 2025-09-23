"""ASR microservice using FastAPI + faster-whisper
   Works-first profile: mic-only, VAD off, stable chunking, proper scaling + resample."""

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
from typing import Dict, Optional, List

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

# ----------------------------- Configuration -----------------------------

# We capture at the device's native sample rate (e.g., 44100 / 48000) and slice by *that* rate.
# Then we normalize and resample to 16000 for Whisper.

TARGET_SR = 16000           # Whisper expects ~16 kHz
CAPTURE_CHANNELS = 1        # mono capture
SAMPLE_WIDTH = 2            # int16 bytes per sample
HOP_SEC = 2.0               # chunk length for stability (can lower later for latency)
READS_PER_SEC = 2           # read loop responsiveness (blocksize = src_sr // READS_PER_SEC)

# Whisper model (GPU preferred; CPU fallback only if CUDA load fails)
MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"
BEAM_SIZE = 5
VAD_FILTER = False          # keep off until everything works
WORD_TIMESTAMPS = True

# Logging
logger = logging.getLogger("asr_service")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


# ----------------------------- Session State -----------------------------

@dataclass
class SessionState:
    session_id: str
    queue: queue_module.Queue = field(default_factory=queue_module.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    finished_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    chunk_index: int = 0
    total_target_frames: int = 0  # running timeline in TARGET_SR units
    active: bool = True


sessions: Dict[str, SessionState] = {}
sessions_lock = threading.Lock()
model_lock = threading.Lock()
MODEL: Optional[WhisperModel] = None
CURRENT_DEVICE = DEVICE
CURRENT_COMPUTE_TYPE = COMPUTE_TYPE


# ----------------------------- Pydantic -----------------------------

class StopSessionRequest(BaseModel):
    session_id: str


# ----------------------------- Model Init -----------------------------

def init_model() -> None:
    """Load Whisper on CUDA; CPU fallback only if needed to avoid total crash."""
    global MODEL, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE
    with model_lock:
        if MODEL is not None:
            return
        try:
            MODEL = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
            CURRENT_DEVICE = DEVICE
            CURRENT_COMPUTE_TYPE = COMPUTE_TYPE
            logger.info("Loaded model '%s' on %s (%s)", MODEL_NAME, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE)
        except Exception as exc:  # noqa: BLE001
            logger.warning("CUDA init failed (%s). Falling back to CPU (slower).", exc)
            MODEL = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
            CURRENT_DEVICE = "cpu"
            CURRENT_COMPUTE_TYPE = "int8"
            logger.info("Loaded model '%s' on %s (%s)", MODEL_NAME, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE)


def close_model() -> None:
    """Release model (lets Python GC free GPU memory)."""
    global MODEL
    with model_lock:
        MODEL = None


# ----------------------------- DSP Helpers -----------------------------

def _downmix_to_mono_int16(frames: np.ndarray) -> np.ndarray:
    """Downmix interleaved int16 frames to mono if needed; return 1-D int16."""
    if frames.ndim == 1:
        return frames
    if frames.shape[1] == 1:
        return frames[:, 0]
    # average L+R using int32 then back to int16
    mixed = (frames[:, 0].astype(np.int32) + frames[:, 1].astype(np.int32)) // 2
    return mixed.astype(np.int16)


def _int16_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 in [-1, 1]."""
    # 32768.0 keeps full-scale int16 at ~1.0
    return x.astype(np.float32) / 32768.0


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear resample using NumPy interpolation (fast and sufficient for ASR)."""
    if x.size == 0 or src_sr == dst_sr:
        return x.astype(np.float32, copy=False)
    x = x.astype(np.float32, copy=False)
    duration = x.shape[0] / float(src_sr)
    new_len = max(1, int(round(duration * dst_sr)))
    if x.shape[0] == 1:
        return np.full((new_len,), x[0], dtype=np.float32)
    xp = np.linspace(0.0, duration, num=x.shape[0], endpoint=False, dtype=np.float32)
    xnew = np.linspace(0.0, duration, num=new_len, endpoint=False, dtype=np.float32)
    y = np.interp(xnew, xp, x).astype(np.float32, copy=False)
    return y


# ----------------------------- Transcription -----------------------------

def _transcribe_chunk(
    session: SessionState,
    chunk_int16: np.ndarray,
    src_sr: int,
    chunk_index: int,
    final_chunk: bool
) -> Optional[dict]:
    """Normalize, resample to 16 kHz, run Whisper, and package an event."""
    if chunk_int16.size == 0:
        return None

    # Ensure mono int16 → float32 [-1,1]
    if chunk_int16.ndim == 2:
        chunk_int16 = _downmix_to_mono_int16(chunk_int16)
    audio_f32 = _int16_to_float32(chunk_int16)

    # Resample to 16 kHz for Whisper
    if src_sr != TARGET_SR:
        audio_f32 = _resample_linear(audio_f32, src_sr=src_sr, dst_sr=TARGET_SR)

    # Quick sanity: RMS should be ~1e-3..1e-1 when speech/media is present
    rms = float(np.sqrt(np.mean(np.square(audio_f32))) + 1e-12)
    logger.info("Chunk #%d RMS: %.6f", chunk_index, rms)

    with model_lock:
        if MODEL is None:
            init_model()
        model = MODEL
    assert model is not None

    segments, info = model.transcribe(
        audio=audio_f32,
        language=None,              # auto-detect
        beam_size=BEAM_SIZE,
        vad_filter=VAD_FILTER,      # keep off for reliability on Windows
        word_timestamps=WORD_TIMESTAMPS,
    )

    text_parts: List[str] = []
    words_payload: List[dict] = []
    seg_start = None
    seg_end = None

    for seg in segments:
        cleaned = (seg.text or "").strip()
        if cleaned:
            text_parts.append(cleaned)
        if seg.words:
            for w in seg.words:
                words_payload.append({
                    "start": round(w.start, 2) if w.start is not None else None,
                    "end": round(w.end, 2) if w.end is not None else None,
                    "word": w.word,
                })
        if seg_start is None and seg.start is not None:
            seg_start = float(seg.start)
        if seg.end is not None:
            seg_end = float(seg.end)

    if seg_start is None:
        seg_start = 0.0
    if seg_end is None:
        seg_end = len(audio_f32) / TARGET_SR

    # Build event timestamps using the running TARGET_SR timeline
    ev = {
        "session_id": session.session_id,
        "chunk_index": chunk_index,
        "start_sec": round((session.total_target_frames / TARGET_SR) + seg_start, 2),
        "end_sec": round((session.total_target_frames / TARGET_SR) + seg_end, 2),
        "text": " ".join(text_parts).strip(),
        "words": words_payload,
        "final": final_chunk,
    }
    return ev


def _capture_and_transcribe(session: SessionState) -> None:
    """Mic-only capture; stable chunking by device rate; clean stop."""
    try:
        # Resolve default input device + its native sample rate
        default_in, _ = sd.default.device
        if default_in is None:
            # Choose first input-capable device if default missing
            devices = sd.query_devices()
            default_in = next((i for i, d in enumerate(devices) if d["max_input_channels"] > 0), None)
        info = sd.query_devices(default_in if default_in is not None else None, "input")
        src_sr = int(info["default_samplerate"])
        name = info.get("name", "Unknown input")
        logger.info("Using input device #%s %s @ %d Hz", default_in if default_in is not None else "auto", name, src_sr)

        # Read size for responsiveness; slice into exact HOP_SEC chunks by *device* rate
        read_frames = max(1, src_sr // READS_PER_SEC)
        hop_frames_src = int(round(HOP_SEC * src_sr))
        hop_bytes_src = hop_frames_src * SAMPLE_WIDTH  # mono int16

        audio_buffer = bytearray()

        with sd.InputStream(
            samplerate=src_sr,
            channels=CAPTURE_CHANNELS,
            dtype="int16",
            blocksize=read_frames,
            device=default_in,
        ) as stream:
            logger.info("Session %s started audio capture on %s/%s", session.session_id, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE)

            while not session.stop_event.is_set():
                frames, overflow = stream.read(read_frames)
                if overflow:
                    logger.warning("Audio overflow on session %s", session.session_id)
                audio_buffer.extend(frames.tobytes())

                while len(audio_buffer) >= hop_bytes_src:
                    raw = bytes(audio_buffer[:hop_bytes_src])
                    del audio_buffer[:hop_bytes_src]
                    chunk = np.frombuffer(raw, dtype=np.int16)  # mono int16
                    # Transcribe this hop
                    ev = _transcribe_chunk(
                        session=session,
                        chunk_int16=chunk,
                        src_sr=src_sr,
                        chunk_index=session.chunk_index,
                        final_chunk=False,
                    )
                    # Advance running timeline by exactly one hop at TARGET_SR
                    session.total_target_frames += int(round(HOP_SEC * TARGET_SR))
                    if ev:
                        session.queue.put(ev)
                    session.chunk_index += 1

            # Flush tail as final (if any)
            if len(audio_buffer) > 0:
                raw = bytes(audio_buffer)
                audio_buffer.clear()
                chunk = np.frombuffer(raw, dtype=np.int16)
                ev = _transcribe_chunk(
                    session=session,
                    chunk_int16=chunk,
                    src_sr=src_sr,
                    chunk_index=session.chunk_index,
                    final_chunk=True,
                )
                # advance timeline by real tail length mapped to TARGET_SR
                tail_sec = len(chunk) / float(src_sr)
                session.total_target_frames += int(round(tail_sec * TARGET_SR))
                if ev:
                    session.queue.put(ev)
                session.chunk_index += 1
            else:
                # Emit final empty event to signal completion
                session.queue.put({
                    "session_id": session.session_id,
                    "chunk_index": session.chunk_index,
                    "start_sec": session.total_target_frames / TARGET_SR,
                    "end_sec": session.total_target_frames / TARGET_SR,
                    "text": "",
                    "words": [],
                    "final": True,
                })

            logger.info("Session %s finished audio capture", session.session_id)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Session %s error: %s", session.session_id, exc)
        session.queue.put({
            "session_id": session.session_id,
            "chunk_index": session.chunk_index,
            "start_sec": session.total_target_frames / TARGET_SR,
            "end_sec": session.total_target_frames / TARGET_SR,
            "text": "",
            "words": [],
            "final": True,
        })
    finally:
        session.queue.put(None)
        session.active = False
        session.finished_event.set()


# ----------------------------- Session Control -----------------------------

def _ensure_single_active_session() -> None:
    with sessions_lock:
        active = [s for s in sessions.values() if s.active and not s.stop_event.is_set()]
        if active:
            raise HTTPException(status_code=409, detail="An ASR session is already running")


def start_new_session() -> SessionState:
    _ensure_single_active_session()
    session_id = str(uuid.uuid4())
    state = SessionState(session_id=session_id)
    with sessions_lock:
        sessions[session_id] = state
    t = threading.Thread(
        target=_capture_and_transcribe,
        args=(state,),
        name=f"asr-session-{session_id}",
        daemon=True,
    )
    state.thread = t
    t.start()
    return state


def stop_session(session_id: str) -> None:
    with sessions_lock:
        s = sessions.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")
    s.stop_event.set()
    logger.info("Stopping session %s on request", session_id)
    if s.thread and s.thread.is_alive():
        s.thread.join(timeout=5)
    s.finished_event.wait(timeout=5)
    logger.info("Session %s fully stopped", session_id)


def _cleanup_session(session_id: str) -> None:
    with sessions_lock:
        sessions.pop(session_id, None)


# ----------------------------- FastAPI App -----------------------------

app = FastAPI(title="ASR Service", version="1.2.0")


@app.on_event("startup")
async def on_startup() -> None:
    init_model()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    with sessions_lock:
        running = list(sessions.values())
    for s in running:
        if s.active and not s.stop_event.is_set():
            s.stop_event.set()
        if s.thread and s.thread.is_alive():
            s.thread.join(timeout=5)
    close_model()


# ----------------------------- Endpoints -----------------------------

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/session/start")
async def session_start() -> JSONResponse:
    s = start_new_session()
    return JSONResponse({"session_id": s.session_id})


@app.post("/session/stop")
async def session_stop(payload: StopSessionRequest) -> JSONResponse:
    stop_session(payload.session_id)
    return JSONResponse({"status": "stopped", "session_id": payload.session_id})


@app.get("/session/{session_id}/stream")
async def session_stream(session_id: str, request: Request) -> StreamingResponse:
    with sessions_lock:
        s = sessions.get(session_id)
        if s is None:
            raise HTTPException(status_code=404, detail="Session not found")
        q = s.queue

    async def gen():
        loop = asyncio.get_running_loop()
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("Client disconnected from session %s stream", session_id)
                    break
                item = await loop.run_in_executor(None, q.get)
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            _cleanup_session(session_id)

    return StreamingResponse(gen(), media_type="text/event-stream")


atexit.register(close_model)

# Run:
# uvicorn asr_service:app --host 127.0.0.1 --port 5001 --reload
