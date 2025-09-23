"""ASR microservice using FastAPI and faster-whisper — GPU, low-latency, proper scaling."""

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

# Audio pipeline
TARGET_SR = 16000            # Whisper expects ~16 kHz
CHANNELS = 1                 # mono
SAMPLE_WIDTH = 2            # bytes per sample (int16)
HOP_SEC = 0.75              # chunk hop for low latency
READ_HZ = 2                 # how many reads per second for responsiveness (affects read block size)

# Model config (GPU)
MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
DEVICE = "cuda"             # stay on GPU; we only fallback if CUDA init fails
COMPUTE_TYPE = "int8_float16"
BEAM_SIZE = 5
USE_VAD = False             # keep False on Windows unless onnxruntime is properly installed
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
    total_frames_processed: int = 0
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
    """Load Whisper on CUDA with the requested compute type; fallback to CPU only if absolutely necessary."""
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
            # As per your request, we want GPU. We still provide a hard fallback to avoid total crash.
            fallback_device = "cpu"
            fallback_compute = "int8"
            logger.warning("Failed to load model on CUDA (%s). Falling back to CPU.", exc)
            MODEL = WhisperModel(MODEL_NAME, device=fallback_device, compute_type=fallback_compute)
            CURRENT_DEVICE = fallback_device
            CURRENT_COMPUTE_TYPE = fallback_compute
            logger.info("Loaded model '%s' on %s (%s)", MODEL_NAME, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE)


def close_model() -> None:
    """Release model (lets Python GC free GPU memory)."""
    global MODEL
    with model_lock:
        MODEL = None


# ----------------------------- DSP Helpers -----------------------------

def _int16_to_float32(x: np.ndarray) -> np.ndarray:
    """Convert int16 PCM to float32 in [-1, 1]."""
    # Avoid division by zero; 32768.0 is safe for int16 full scale
    return (x.astype(np.float32)) / 32768.0


def _downmix_stereo_to_mono(frames: np.ndarray) -> np.ndarray:
    """If frames are interleaved stereo int16, downmix to mono."""
    if frames.ndim == 1:
        return frames
    # frames shape likely (n_frames, channels)
    if frames.shape[1] == 1:
        return frames[:, 0]
    # average L+R safely using int32 then back to int16
    mixed = (frames[:, 0].astype(np.int32) + frames[:, 1].astype(np.int32)) // 2
    return mixed.astype(np.int16)


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Lightweight linear resampler using NumPy interpolation (good enough for ASR)."""
    if src_sr == dst_sr or x.size == 0:
        return x
    # Ensure 1D float32 for interpolation
    x = x.astype(np.float32, copy=False)
    duration = x.shape[0] / float(src_sr)
    new_len = max(1, int(round(duration * dst_sr)))
    # Avoid degenerate case
    if x.shape[0] == 1:
        return np.full((new_len,), x[0], dtype=np.float32)
    xp = np.linspace(0.0, duration, num=x.shape[0], endpoint=False, dtype=np.float32)
    fp = x
    xnew = np.linspace(0.0, duration, num=new_len, endpoint=False, dtype=np.float32)
    y = np.interp(xnew, xp, fp).astype(np.float32, copy=False)
    return y


# ----------------------------- Transcription -----------------------------

def _transcribe_chunk(session: SessionState, chunk_int16: np.ndarray, chunk_index: int, src_sr: int, final_chunk: bool) -> Optional[dict]:
    """Scale, resample, and transcribe a single chunk."""
    if chunk_int16.size == 0:
        return None

    # If 2-ch, downmix; ensure 1-D int16
    if chunk_int16.ndim == 2:
        chunk_int16 = _downmix_stereo_to_mono(chunk_int16)

    # Scale to [-1, 1]
    audio_float = _int16_to_float32(chunk_int16)

    # Resample to 16 kHz for Whisper
    if src_sr != TARGET_SR:
        audio_float = _resample_linear(audio_float, src_sr=src_sr, dst_sr=TARGET_SR)

    # Simple RMS for sanity: expected ~1e-3 .. 1e-1 while speaking or for media with normal loudness
    rms = float(np.sqrt(np.mean(np.square(audio_float))) + 1e-12)
    logger.info("Audio RMS: %.6f (chunk #%d)", rms, chunk_index)

    with model_lock:
        if MODEL is None:
            init_model()
        model = MODEL
    assert model is not None

    # Transcribe — VAD off for Windows unless onnxruntime is set up
    segments, info = model.transcribe(
        audio=audio_float,
        language=None,               # auto-detect
        beam_size=BEAM_SIZE,
        vad_filter=USE_VAD,
        word_timestamps=WORD_TIMESTAMPS,
    )

    text_parts: List[str] = []
    words_payload: List[dict] = []
    start_time = None
    end_time = None

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
        if start_time is None and seg.start is not None:
            start_time = float(seg.start)
        if seg.end is not None:
            end_time = float(seg.end)

    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = len(audio_float) / TARGET_SR

    # Map local times into running timeline (based on frames processed at TARGET_SR)
    # total_frames_processed tracks input frames at *source* rate; convert to seconds consistently
    # We approximate by counting seconds in TARGET_SR units.
    chunk_start_sec = session.total_frames_processed / float(TARGET_SR)

    event = {
        "session_id": session.session_id,
        "chunk_index": chunk_index,
        "start_sec": round(chunk_start_sec + start_time, 2),
        "end_sec": round(chunk_start_sec + min(end_time, len(audio_float) / TARGET_SR), 2),
        "text": " ".join(text_parts).strip(),
        "words": words_payload,
        "final": final_chunk,
    }
    return event


def _capture_and_transcribe(session: SessionState) -> None:
    """
    Capture from default input (Windows WASAPI mic by default),
    push normalized, resampled chunks to Whisper, stream results as SSE.
    """
    # We will open the *default* input device. On Windows WASAPI, this will be your mic if default.
    # For system-audio capture, you need a loopback device; we can add selection later if needed.
    try:
        # Determine device and sampling rate
        default_in, _ = sd.default.device
        if default_in is None:
            # Fallback to first input-capable device
            devices = sd.query_devices()
            default_in = next((i for i, d in enumerate(devices) if d["max_input_channels"] > 0), None)
        device_info = sd.query_devices(default_in) if default_in is not None else sd.query_devices(None, "input")
        src_sr = int(device_info["default_samplerate"])
        logger.info("Using input device #%s %s @ %d Hz",
                    default_in if default_in is not None else "auto",
                    device_info["name"], src_sr)

        # Sizes
        hop_frames_src = int(round(HOP_SEC * src_sr))
        read_frames = max(1, src_sr // READ_HZ)

        # We collect raw bytes then slice into fixed hops for consistent latency
        chunk_bytes_target = int(round(HOP_SEC * TARGET_SR)) * SAMPLE_WIDTH  # target hop sized in 16kHz samples
        audio_buffer = bytearray()

        # Open input stream
        with sd.InputStream(
            samplerate=src_sr,
            channels=CHANNELS,
            dtype="int16",
            blocksize=read_frames,
            device=default_in,
        ) as stream:
            logger.info("Session %s started audio capture on %s/%s", session.session_id, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE)

            # We will still hop by *time*, not by incoming block count; accumulate then cut consistent sub-chunks
            while not session.stop_event.is_set():
                frames, overflowed = stream.read(read_frames)
                if overflowed:
                    logger.warning("Audio input overflow detected for session %s", session.session_id)
                audio_buffer.extend(frames.tobytes())

                # Whenever buffer has at least TARGET hop worth of *target* samples, cut one chunk
                while len(audio_buffer) >= chunk_bytes_target:
                    raw_chunk = bytes(audio_buffer[:chunk_bytes_target])
                    del audio_buffer[:chunk_bytes_target]
                    # Interpret at source rate; we will resample inside _transcribe_chunk
                    # Note: even though chunk_bytes_target is sized for 16kHz, we treat it as a time window.
                    # The int16 buffer content represents 'HOP_SEC' seconds worth of samples at *source* pacing.
                    chunk_np = np.frombuffer(raw_chunk, dtype=np.int16)

                    event = _transcribe_chunk(session, chunk_np, session.chunk_index, src_sr=src_sr, final_chunk=False)
                    # Increment timeline in TARGET_SR domain
                    session.total_frames_processed += int(round(HOP_SEC * TARGET_SR))
                    if event:
                        session.queue.put(event)
                    session.chunk_index += 1

            # Flush any tail as final
            if audio_buffer:
                raw_chunk = bytes(audio_buffer)
                audio_buffer.clear()
                chunk_np = np.frombuffer(raw_chunk, dtype=np.int16)
                event = _transcribe_chunk(session, chunk_np, session.chunk_index, src_sr=src_sr, final_chunk=True)
                session.total_frames_processed += int(round((len(chunk_np) / float(src_sr)) * TARGET_SR))
                if event:
                    session.queue.put(event)
                session.chunk_index += 1
            else:
                # Emit final empty event to signal completion if nothing captured post-stop
                final_event = {
                    "session_id": session.session_id,
                    "chunk_index": session.chunk_index,
                    "start_sec": session.total_frames_processed / float(TARGET_SR),
                    "end_sec": session.total_frames_processed / float(TARGET_SR),
                    "text": "",
                    "words": [],
                    "final": True,
                }
                session.queue.put(final_event)

            logger.info("Session %s finished audio capture", session.session_id)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Session %s encountered an error: %s", session.session_id, exc)
        error_event = {
            "session_id": session.session_id,
            "chunk_index": session.chunk_index,
            "start_sec": session.total_frames_processed / float(TARGET_SR),
            "end_sec": session.total_frames_processed / float(TARGET_SR),
            "text": "",
            "words": [],
            "final": True,
        }
        session.queue.put(error_event)
    finally:
        session.queue.put(None)
        session.active = False
        session.finished_event.set()


# ----------------------------- Session Control -----------------------------

def _ensure_single_active_session() -> None:
    with sessions_lock:
        active_sessions = [s for s in sessions.values() if s.active and not s.stop_event.is_set()]
        if active_sessions:
            raise HTTPException(status_code=409, detail="An ASR session is already running")


def start_new_session() -> SessionState:
    _ensure_single_active_session()
    session_id = str(uuid.uuid4())
    session = SessionState(session_id=session_id)
    with sessions_lock:
        sessions[session_id] = session
    thread = threading.Thread(target=_capture_and_transcribe, args=(session,), name=f"asr-session-{session_id}", daemon=True)
    session.thread = thread
    thread.start()
    return session


def stop_session(session_id: str) -> None:
    with sessions_lock:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

    logger.info("Stopping session %s on request", session_id)
    session.stop_event.set()
    if session.thread and session.thread.is_alive():
        session.thread.join(timeout=5)
    session.finished_event.wait(timeout=5)
    logger.info("Session %s fully stopped and cleaned up", session_id)


# ----------------------------- FastAPI App -----------------------------

app = FastAPI(title="ASR Service", version="1.1.0")


@app.on_event("startup")
async def on_startup() -> None:
    init_model()


@app.on_event("shutdown")
async def on_shutdown() -> None:
    with sessions_lock:
        running_sessions = list(sessions.values())
    for session in running_sessions:
        if session.active and not session.stop_event.is_set():
            session.stop_event.set()
        if session.thread and session.thread.is_alive():
            session.thread.join(timeout=5)
    close_model()


def _cleanup_session(session_id: str) -> None:
    with sessions_lock:
        sessions.pop(session_id, None)


# ----------------------------- Endpoints -----------------------------

@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/session/start")
async def session_start() -> JSONResponse:
    session = start_new_session()
    return JSONResponse({"session_id": session.session_id})


@app.post("/session/stop")
async def session_stop(payload: StopSessionRequest) -> JSONResponse:
    stop_session(payload.session_id)
    return JSONResponse({"status": "stopped", "session_id": payload.session_id})


@app.get("/session/{session_id}/stream")
async def session_stream(session_id: str, request: Request) -> StreamingResponse:
    with sessions_lock:
        session = sessions.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")
        session_queue = session.queue

    async def event_generator():
        loop = asyncio.get_running_loop()
        try:
            while True:
                if await request.is_disconnected():
                    logger.info("Client disconnected from session %s stream", session_id)
                    break
                item = await loop.run_in_executor(None, session_queue.get)
                if item is None:
                    break
                yield f"data: {json.dumps(item)}\n\n"
        finally:
            _cleanup_session(session_id)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


atexit.register(close_model)

# How to run:
# uvicorn asr_service:app --host 127.0.0.1 --port 5001 --reload
