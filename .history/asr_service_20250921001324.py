"""ASR microservice using FastAPI + faster-whisper
   Works-first profile with rolling context:
   - CUDA preferred (CPU fallback only if CUDA init fails)
   - Mic-only capture (system audio can be added later)
   - No VAD (avoids onnxruntime issues)
   - 8s rolling buffer, 2s hop → reliable tokens
   - Emits only NEW text delta each hop (UI-friendly)
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
from typing import Dict, Optional, List, Tuple

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

# ----------------------------- Configuration -----------------------------

TARGET_SR = 16000              # Whisper works best ~16k
CAPTURE_CHANNELS = 1           # mono
SAMPLE_WIDTH = 2               # int16
HOP_SEC = 2.0                  # inference stride
CTX_SEC = 8.0                  # rolling window length
READS_PER_SEC = 2              # stream.read() rate for responsiveness

MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"
BEAM_SIZE = 5
VAD_FILTER = False             # keep off for stability on Windows
WORD_TIMESTAMPS = True
TEMPERATURE = 0.0              # stable decoding

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
    # running timeline in TARGET_SR frames
    total_target_frames: int = 0
    # rolling buffer (TARGET_SR)
    ctx_buffer: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    # last emitted text to compute suffix
    last_emitted: str = ""
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
    """Load Whisper on CUDA; CPU fallback if init fails (slow, but prevents crash)."""
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
    global MODEL
    with model_lock:
        MODEL = None


# ----------------------------- DSP Helpers -----------------------------

def _downmix_to_mono_int16(frames: np.ndarray) -> np.ndarray:
    if frames.ndim == 1:
        return frames
    if frames.shape[1] == 1:
        return frames[:, 0]
    mixed = (frames[:, 0].astype(np.int32) + frames[:, 1].astype(np.int32)) // 2
    return mixed.astype(np.int16)


def _int16_to_float32(x: np.ndarray) -> np.ndarray:
    # Use 32768 for symmetric scaling (keeps FS ≈ 1.0)
    return x.astype(np.float32) / 32768.0


def _resample_linear(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
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


# ----------------------------- Text Delta -----------------------------

def _suffix_delta(prev: str, curr: str) -> str:
    """Return the new suffix added to curr after prev; simple LCP diff."""
    if not prev:
        return curr
    # find common prefix length
    i = 0
    m = min(len(prev), len(curr))
    while i < m and prev[i] == curr[i]:
        i += 1
    # if prev was a prefix of curr, emit suffix; else (edit), emit curr
    if prev == curr[:len(prev)]:
        return curr[len(prev):]
    return curr


# ----------------------------- Transcription -----------------------------

def _run_whisper(audio_f32: np.ndarray) -> Tuple[str, List[dict], float, float]:
    """Run Whisper over audio_f32@16k. Return (full_text, words[], start, end)."""
    with model_lock:
        if MODEL is None:
            init_model()
        model = MODEL
    assert model is not None

    segments, info = model.transcribe(
        audio=audio_f32,
        language=None,
        beam_size=BEAM_SIZE,
        vad_filter=VAD_FILTER,
        word_timestamps=WORD_TIMESTAMPS,
        temperature=TEMPERATURE,
        condition_on_previous_text=True,
    )

    text_parts: List[str] = []
    words_payload: List[dict] = []
    seg_start = None
    seg_end = None

    for seg in segments:
        txt = (seg.text or "").strip()
        if txt:
            text_parts.append(txt)
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

    return " ".join(text_parts).strip(), words_payload, seg_start, seg_end


def _transcribe_with_context(session: SessionState, src_block_f32_16k: np.ndarray, is_final: bool) -> Optional[dict]:
    """Append to rolling buffer, run inference every hop, emit suffix delta."""
    # append to rolling buffer
    session.ctx_buffer = np.concatenate([session.ctx_buffer, src_block_f32_16k], dtype=np.float32)
    # trim to CTX_SEC
    max_ctx_frames = int(round(CTX_SEC * TARGET_SR))
    if session.ctx_buffer.size > max_ctx_frames:
        session.ctx_buffer = session.ctx_buffer[-max_ctx_frames:]

    # compute RMS for sanity
    rms = float(np.sqrt(np.mean(np.square(src_block_f32_16k))) + 1e-12)
    logger.info("Chunk #%d RMS: %.6f", session.chunk_index, rms)

    # Run whisper on the whole context buffer
    full_text, words, seg_start, seg_end = _run_whisper(session.ctx_buffer)
    if not full_text:
        # No tokens → skip emitting to avoid “…” in UI
        return None

    # Compute delta vs last emitted
    delta = _suffix_delta(session.last_emitted, full_text).strip()
    if not delta:
        # No new text → nothing to emit
        return None

    # Update last emitted
    session.last_emitted = full_text

    # Align times: context buffer ends at current timeline end (session.total_target_frames)
    ctx_len_sec = session.ctx_buffer.size / float(TARGET_SR)
    # Map seg_start/seg_end within context to absolute timeline:
    abs_start = (session.total_target_frames / TARGET_SR) - ctx_len_sec + seg_start
    abs_end = (session.total_target_frames / TARGET_SR) - ctx_len_sec + seg_end

    event = {
        "session_id": session.session_id,
        "chunk_index": session.chunk_index,
        "start_sec": round(max(0.0, abs_start), 2),
        "end_sec": round(max(0.0, abs_end), 2),
        "text": delta,
        "words": words,  # words are relative to context; acceptable for display
        "final": is_final,
    }
    return event


def _capture_and_transcribe(session: SessionState) -> None:
    try:
        # Pick default input device + native rate
        default_in, _ = sd.default.device
        if default_in is None:
            devices = sd.query_devices()
            default_in = next((i for i, d in enumerate(devices) if d["max_input_channels"] > 0), None)
        info = sd.query_devices(default_in if default_in is not None else None, "input")
        src_sr = int(info["default_samplerate"])
        name = info.get("name", "Unknown input")
        logger.info("Using input device #%s %s @ %d Hz", default_in if default_in is not None else "auto", name, src_sr)

        # read granularity and hop sizing in *source* rate
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
                    # device mono int16 → f32 → resample to 16k
                    block_i16 = np.frombuffer(raw, dtype=np.int16)
                    if block_i16.ndim == 2:
                        block_i16 = _downmix_to_mono_int16(block_i16)
                    block_f32 = _int16_to_float32(block_i16)
                    block_16k = _resample_linear(block_f32, src_sr=src_sr, dst_sr=TARGET_SR)

                    # transcribe with context and emit delta (non-final)
                    ev = _transcribe_with_context(session, block_16k, is_final=False)

                    # advance running timeline by hop
                    session.total_target_frames += int(round(HOP_SEC * TARGET_SR))

                    if ev:
                        session.queue.put(ev)
                    session.chunk_index += 1

            # flush tail as final
            if len(audio_buffer) > 0:
                raw = bytes(audio_buffer)
                audio_buffer.clear()
                tail_i16 = np.frombuffer(raw, dtype=np.int16)
                if tail_i16.ndim == 2:
                    tail_i16 = _downmix_to_mono_int16(tail_i16)
                tail_f32 = _int16_to_float32(tail_i16)
                tail_16k = _resample_linear(tail_f32, src_sr=src_sr, dst_sr=TARGET_SR)

                ev = _transcribe_with_context(session, tail_16k, is_final=True)
                tail_sec = len(tail_16k) / float(TARGET_SR)
                session.total_target_frames += int(round(tail_sec * TARGET_SR))
                if ev:
                    session.queue.put(ev)
                session.chunk_index += 1
            else:
                # emit a final empty to close SSE consumers gracefully
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
    logger.info("Session %s fully stopped and cleaned up", session_id)


def _cleanup_session(session_id: str) -> None:
    with sessions_lock:
        sessions.pop(session_id, None)


# ----------------------------- FastAPI App -----------------------------

app = FastAPI(title="ASR Service", version="1.3.0")


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
