"""ASR microservice using FastAPI and faster-whisper with safe VAD fallback."""
import asyncio
import atexit
import json
import logging
import os
import queue as queue_module
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, AsyncGenerator

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

# ------------------------ Configuration ------------------------
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per sample (16-bit PCM)
CHUNK_SEC = 30
MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"  # falls back to CPU/int8 if CUDA unavailable
BEAM_SIZE = 5
VAD_FILTER_DEFAULT = True          # preferred default
WORD_TIMESTAMPS = True

# Check VAD availability (onnxruntime)
try:
    import onnxruntime  # noqa: F401
    HAS_ONNXRUNTIME = True
except Exception:
    HAS_ONNXRUNTIME = False

logger = logging.getLogger("asr_service")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


# ------------------------ Session State ------------------------
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


class StopSessionRequest(BaseModel):
    session_id: str


# ------------------------ Model lifecycle ------------------------
def init_model() -> None:
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
            fallback_device = "cpu"
            fallback_compute = "int8"
            logger.warning("Failed to load model on CUDA (%s). Falling back to CPU.", exc)
            MODEL = WhisperModel(MODEL_NAME, device=fallback_device, compute_type=fallback_compute)
            CURRENT_DEVICE = fallback_device
            CURRENT_COMPUTE_TYPE = fallback_compute
            logger.info("Loaded model '%s' on %s (%s)", MODEL_NAME, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE)


def close_model() -> None:
    global MODEL
    with model_lock:
        MODEL = None


# ------------------------ Transcription ------------------------
def _run_transcribe(model: WhisperModel, audio_float: np.ndarray, want_vad: bool):
    """Call faster-whisper with optional VAD, retry once without VAD if runtime demands onnxruntime."""
    use_vad = bool(want_vad and HAS_ONNXRUNTIME)
    try:
        return model.transcribe(
            audio_float,
            beam_size=BEAM_SIZE,
            vad_filter=use_vad,
            word_timestamps=WORD_TIMESTAMPS,
        )
    except RuntimeError as e:
        # Typical message: "Applying the VAD filter requires the onnxruntime package"
        if "onnxruntime" in str(e).lower() or "vad" in str(e).lower():
            logger.warning("VAD unavailable (%s). Retrying transcribe without VAD.", e)
            return model.transcribe(
                audio_float,
                beam_size=BEAM_SIZE,
                vad_filter=False,
                word_timestamps=WORD_TIMESTAMPS,
            )
        raise
    except TypeError as e:
        # Safety for unexpected kw issues
        logger.error("WhisperModel.transcribe() call failed: %s", e)
        return [], None


def _transcribe_chunk(session: SessionState, chunk_frames: np.ndarray, chunk_index: int, final_chunk: bool) -> Optional[dict]:
    if chunk_frames.size == 0:
        return None

    with model_lock:
        if MODEL is None:
            init_model()
        model = MODEL
    assert model is not None

    # int16 -> float32 in [-1, 1]
    audio_float = chunk_frames.astype(np.float32) / np.iinfo(np.int16).max

    # Prefer VAD but auto-disable if onnxruntime is missing
    if VAD_FILTER_DEFAULT and not HAS_ONNXRUNTIME:
        logger.debug("VAD requested but onnxruntime not found; using no-VAD for this chunk.")

    segments, info = _run_transcribe(model, audio_float, VAD_FILTER_DEFAULT)

    text_parts = []
    words_payload = []
    start_time = None
    end_time = None
    for segment in segments:
        cleaned = segment.text.strip()
        if cleaned:
            text_parts.append(cleaned)
        if segment.words:
            for word in segment.words:
                words_payload.append({
                    "start": round(word.start, 2) if word.start is not None else None,
                    "end": round(word.end, 2) if word.end is not None else None,
                    "word": word.word,
                })
        if start_time is None and segment.start is not None:
            start_time = segment.start
        if segment.end is not None:
            end_time = segment.end

    if start_time is None:
        start_time = 0.0
    if end_time is None:
        end_time = chunk_frames.size / SAMPLE_RATE

    chunk_start_sec = session.total_frames_processed / SAMPLE_RATE

    event = {
        "session_id": session.session_id,
        "chunk_index": chunk_index,
        "start_sec": round(chunk_start_sec + start_time, 2),
        "end_sec": round(chunk_start_sec + min(end_time, chunk_frames.size / SAMPLE_RATE), 2),
        "text": " ".join(text_parts).strip(),
        "words": words_payload,
        "final": final_chunk,
    }
    return event


def _capture_and_transcribe(session: SessionState) -> None:
    chunk_bytes = SAMPLE_RATE * CHUNK_SEC * SAMPLE_WIDTH
    read_frames = SAMPLE_RATE // 2  # 0.5 second reads
    audio_buffer = bytearray()
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            dtype="int16",
            blocksize=0,
        ) as stream:
            logger.info(
                "Session %s started audio capture on %s/%s",
                session.session_id, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE
            )
            while not session.stop_event.is_set():
                frames, overflowed = stream.read(read_frames)
                if overflowed:
                    logger.warning("Audio input overflow detected for session %s", session.session_id)
                audio_buffer.extend(frames.tobytes())

                while len(audio_buffer) >= chunk_bytes:
                    raw_chunk = bytes(audio_buffer[:chunk_bytes])
                    del audio_buffer[:chunk_bytes]
                    chunk_np = np.frombuffer(raw_chunk, dtype=np.int16)
                    event = _transcribe_chunk(session, chunk_np, session.chunk_index, final_chunk=False)
                    session.total_frames_processed += chunk_np.size
                    if event:
                        session.queue.put(event)
                    session.chunk_index += 1

            # Flush remaining
            if audio_buffer:
                raw_chunk = bytes(audio_buffer)
                audio_buffer.clear()
                chunk_np = np.frombuffer(raw_chunk, dtype=np.int16)
                event = _transcribe_chunk(session, chunk_np, session.chunk_index, final_chunk=True)
                session.total_frames_processed += chunk_np.size
                if event:
                    session.queue.put(event)
                session.chunk_index += 1
            else:
                # Signal end
                final_event = {
                    "session_id": session.session_id,
                    "chunk_index": session.chunk_index,
                    "start_sec": session.total_frames_processed / SAMPLE_RATE,
                    "end_sec": session.total_frames_processed / SAMPLE_RATE,
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
            "start_sec": session.total_frames_processed / SAMPLE_RATE,
            "end_sec": session.total_frames_processed / SAMPLE_RATE,
            "text": "",
            "words": [],
            "final": True,
        }
        session.queue.put(error_event)
    finally:
        session.queue.put(None)
        session.active = False
        session.finished_event.set()


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
    session.stop_event.set()
    if session.thread and session.thread.is_alive():
        session.thread.join()
    session.finished_event.wait(timeout=5)


# ------------------------ FastAPI ------------------------
app = FastAPI(title="ASR Service", version="1.0.2")


@app.on_event("startup")
async def on_startup() -> None:
    init_model()
    if VAD_FILTER_DEFAULT and not HAS_ONNXRUNTIME:
        logger.warning("VAD requested but onnxruntime is not installed; running without VAD automatically.")


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

    async def event_generator() -> AsyncGenerator[str, None]:
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

# Run:
# uvicorn asr_service:app --host 127.0.0.1 --port 5001 --reload
