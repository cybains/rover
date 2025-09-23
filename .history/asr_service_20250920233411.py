"""ASR microservice using FastAPI and faster-whisper (Windows-optimized, auto-resample)."""
import asyncio
import atexit
import json
import logging
import os
import queue as queue_module
import threading
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

# =======================
# Configuration (tuned)
# =======================
TARGET_SR = 16000              # Whisper expects ~16 kHz for np.array input
CHANNELS = 1                   # we output mono to ASR
SAMPLE_WIDTH = 2               # 16-bit PCM
CHUNK_SEC = 0.75               # ~0.75 s low-latency chunks

# Model / GPU
MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
DEVICE = "cuda"                # keep GPU as requested
COMPUTE_TYPE = "int8_float16"

# Beam search etc.
BEAM_SIZE = 5
VAD_FILTER = False             # VAD off for instant UX (no onnxruntime needed)
WORD_TIMESTAMPS = True

# Device selection
ASR_INPUT_DEVICE_ENV = os.getenv("ASR_INPUT_DEVICE")  # index or name substring
PREFERRED_HOSTAPI = "Windows WASAPI"
MIC_NAME_HINTS = ["mikrofon", "microphone", "array"]  # case-insensitive

logger = logging.getLogger("asr_service")
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")


@dataclass
class SessionState:
    session_id: str
    queue: queue_module.Queue = field(default_factory=queue_module.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    finished_event: threading.Event = field(default_factory=threading.Event)
    thread: Optional[threading.Thread] = None
    chunk_index: int = 0
    total_target_samples: int = 0  # number of 16kHz mono samples processed so far
    active: bool = True


sessions: Dict[str, SessionState] = {}
sessions_lock = threading.Lock()
model_lock = threading.Lock()
MODEL: Optional[WhisperModel] = None
CURRENT_DEVICE = DEVICE
CURRENT_COMPUTE_TYPE = COMPUTE_TYPE


class StopSessionRequest(BaseModel):
    session_id: str


def init_model() -> None:
    """Load Whisper model on GPU (no CPU fallback, per your requirement)."""
    global MODEL, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE
    with model_lock:
        if MODEL is not None:
            return
        MODEL = WhisperModel(MODEL_NAME, device=DEVICE, compute_type=COMPUTE_TYPE)
        CURRENT_DEVICE = DEVICE
        CURRENT_COMPUTE_TYPE = COMPUTE_TYPE
        logger.info("Loaded model '%s' on %s (%s)", MODEL_NAME, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE)


def close_model() -> None:
    global MODEL
    with model_lock:
        MODEL = None


def _pick_input_device() -> Tuple[Optional[int], str]:
    """
    Prefer Windows WASAPI input. Allow override by ASR_INPUT_DEVICE (index or substring).
    Returns: (device_index, label) or (None, '[none]').
    """
    try:
        hostapis = sd.query_hostapis()
        hostapi_names = {i: ha.get("name", "") for i, ha in enumerate(hostapis)}
    except Exception as e:
        logger.warning("Unable to query hostapis: %s", e)
        hostapi_names = {}

    devices = sd.query_devices()
    default_in, _ = sd.default.device
    chosen_idx: Optional[int] = None
    chosen_label = ""

    # 1) Env override
    if ASR_INPUT_DEVICE_ENV:
        val = ASR_INPUT_DEVICE_ENV.strip()
        try:
            idx = int(val)
            info = sd.query_devices(idx)
            if info["max_input_channels"] > 0:
                chosen_idx = idx
                chosen_label = f"[env index {idx}] {info['name']} ({hostapi_names.get(info['hostapi'], '')})"
                return chosen_idx, chosen_label
        except Exception:
            val_l = val.lower()
            for i, info in enumerate(devices):
                if info.get("max_input_channels", 0) <= 0:
                    continue
                name_l = str(info.get("name", "")).lower()
                if val_l in name_l:
                    chosen_idx = i
                    chosen_label = f"[env match '{val}'] {info['name']} ({hostapi_names.get(info['hostapi'], '')})"
                    return chosen_idx, chosen_label
        logger.warning("ASR_INPUT_DEVICE='%s' not found/usable; falling back to auto.", ASR_INPUT_DEVICE_ENV)

    # 2) WASAPI mic with a hint in name
    for i, info in enumerate(devices):
        if info.get("max_input_channels", 0) <= 0:
            continue
        host = hostapi_names.get(info.get("hostapi", -1), "")
        if PREFERRED_HOSTAPI.lower() in host.lower():
            name_l = str(info.get("name", "")).lower()
            if any(k in name_l for k in MIC_NAME_HINTS):
                chosen_idx = i
                chosen_label = f"[auto WASAPI+mic] {info['name']} ({host})"
                return chosen_idx, chosen_label

    # 2b) Any WASAPI input
    for i, info in enumerate(devices):
        if info.get("max_input_channels", 0) <= 0:
            continue
        host = hostapi_names.get(info.get("hostapi", -1), "")
        if PREFERRED_HOSTAPI.lower() in host.lower():
            chosen_idx = i
            chosen_label = f"[auto WASAPI] {info['name']} ({host})"
            return chosen_idx, chosen_label

    # 3) Default input
    try:
        info = sd.query_devices(default_in)
        if info.get("max_input_channels", 0) > 0:
            chosen_idx = default_in
            chosen_label = f"[default] {info['name']} ({hostapi_names.get(info['hostapi'], '')})"
            return chosen_idx, chosen_label
    except Exception:
        pass

    return None, "[none]"


def _downmix_to_mono_int16(x: np.ndarray) -> np.ndarray:
    """x: np.int16, shape (n,) or (n, ch). Returns int16 mono."""
    if x.ndim == 1:
        return x
    # average across channels with int16 range protection
    mono = x.astype(np.int32).mean(axis=1)
    mono = np.clip(mono, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return mono.astype(np.int16)


def _resample_linear(x: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Fast linear resample: int16/float32 mono -> float32 mono at dst_rate."""
    if src_rate == dst_rate or x.size == 0:
        return x.astype(np.float32)
    n_in = x.shape[0]
    n_out = max(1, int(round(n_in * dst_rate / src_rate)))
    xp = np.linspace(0, n_in, num=n_in, endpoint=False)
    fp = x.astype(np.float32)
    x_new = np.linspace(0, n_in, num=n_out, endpoint=False)
    y = np.interp(x_new, xp, fp).astype(np.float32)
    return y


def _transcribe_chunk(
    session: SessionState,
    mono_16k_float: np.ndarray,
    chunk_index: int,
    final_chunk: bool,
    chunk_start_sec: float,
) -> Optional[dict]:
    """
    mono_16k_float: float32 array at 16 kHz (already resampled).
    chunk_start_sec: time offset (seconds) of this chunk in the session timeline.
    """
    if mono_16k_float.size == 0:
        # still emit final empty if needed
        if final_chunk:
            return {
                "session_id": session.session_id,
                "chunk_index": chunk_index,
                "start_sec": round(chunk_start_sec, 2),
                "end_sec": round(chunk_start_sec, 2),
                "text": "",
                "words": [],
                "final": True,
            }
        return None

    # Early-out for near-silence
    rms = float(np.sqrt(np.mean(np.square(mono_16k_float))) + 1e-9)
    if rms < 1e-4 and not final_chunk:
        return None

    with model_lock:
        if MODEL is None:
            init_model()
        model = MODEL
    assert model is not None

    segments, info = model.transcribe(
        audio=mono_16k_float,
        language=None,              # auto-detect
        beam_size=BEAM_SIZE,
        vad_filter=VAD_FILTER,
        word_timestamps=WORD_TIMESTAMPS,
    )

    text_parts = []
    words_payload = []
    seg_rel_start = None
    seg_rel_end = None

    for segment in segments:
        cleaned = segment.text.strip()
        if cleaned:
            text_parts.append(cleaned)
        if segment.words:
            for word in segment.words:
                words_payload.append({
                    "start": round(chunk_start_sec + (word.start or 0.0), 2) if word.start is not None else None,
                    "end": round(chunk_start_sec + (word.end or 0.0), 2) if word.end is not None else None,
                    "word": word.word,
                })
        if seg_rel_start is None and segment.start is not None:
            seg_rel_start = segment.start
        if segment.end is not None:
            seg_rel_end = segment.end

    if seg_rel_start is None:
        seg_rel_start = 0.0
    if seg_rel_end is None:
        seg_rel_end = mono_16k_float.size / TARGET_SR

    event = {
        "session_id": session.session_id,
        "chunk_index": chunk_index,
        "start_sec": round(chunk_start_sec + seg_rel_start, 2),
        "end_sec": round(chunk_start_sec + seg_rel_end, 2),
        "text": " ".join(text_parts).strip(),
        "words": words_payload,
        "final": final_chunk,
    }
    return event


def _capture_and_transcribe(session: SessionState) -> None:
    """Audio capture loop with device-native rate and on-the-fly resample to 16 kHz."""
    dev_idx, dev_label = _pick_input_device()
    if dev_idx is None:
        logger.error("No usable input device found. Check Windows Sound settings and privacy.")
        # signal final to client
        session.queue.put({
            "session_id": session.session_id,
            "chunk_index": session.chunk_index,
            "start_sec": 0.0,
            "end_sec": 0.0,
            "text": "",
            "words": [],
            "final": True,
        })
        session.queue.put(None)
        session.active = False
        session.finished_event.set()
        return

    info = sd.query_devices(dev_idx)
    hostapi_name = sd.query_hostapis()[info["hostapi"]]["name"]
    device_sr = int(round(info.get("default_samplerate") or 48000))
    device_sr = max(8000, min(192000, device_sr))  # clamp to sane range

    logger.info("Using input device #%s %s @ %d Hz", dev_idx, f"{dev_label}", device_sr)

    # sizes at device rate
    chunk_samples_dev = int(device_sr * CHUNK_SEC)
    read_frames = int(device_sr * 0.5)  # ~0.5 s

    # byte buffer (int16, possibly interleaved channels)
    audio_buffer = bytearray()
    bytes_per_frame_dev = SAMPLE_WIDTH * max(1, min(info["max_input_channels"], 2))  # we request mono; device may still hand us stereo

    try:
        with sd.InputStream(
            device=dev_idx,
            samplerate=device_sr,    # open at native device rate
            channels=min(info["max_input_channels"], 2),  # request up to stereo; we'll downmix
            dtype="int16",
            blocksize=0,
            latency="low",
        ) as stream:
            logger.info("Session %s started audio capture on %s/%s", session.session_id, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE)
            while not session.stop_event.is_set():
                frames, overflowed = stream.read(read_frames)
                if overflowed:
                    logger.warning("Audio input overflow detected for session %s", session.session_id)
                # frames shape: (read_frames, ch)
                audio_buffer.extend(frames.tobytes())

                while len(audio_buffer) >= chunk_samples_dev * bytes_per_frame_dev:
                    raw = bytes(audio_buffer[: chunk_samples_dev * bytes_per_frame_dev])
                    del audio_buffer[: chunk_samples_dev * bytes_per_frame_dev]

                    # int16 -> shape (n, ch)
                    arr = np.frombuffer(raw, dtype=np.int16)
                    ch = min(info["max_input_channels"], 2)
                    if ch > 1:
                        arr = arr.reshape(-1, ch)
                    mono_int16 = _downmix_to_mono_int16(arr)

                    # resample to 16 kHz float32
                    mono_16k_float = _resample_linear(mono_int16, device_sr, TARGET_SR)

                    # compute timeline positions (in TARGET_SR domain)
                    chunk_start_sec = session.total_target_samples / TARGET_SR
                    session.total_target_samples += mono_16k_float.size

                    event = _transcribe_chunk(
                        session=session,
                        mono_16k_float=mono_16k_float,
                        chunk_index=session.chunk_index,
                        final_chunk=False,
                        chunk_start_sec=chunk_start_sec,
                    )
                    if event:
                        session.queue.put(event)
                    session.chunk_index += 1

            # flush remainder as final
            if len(audio_buffer) > 0:
                # pad to frame multiple
                leftover = len(audio_buffer) // bytes_per_frame_dev * bytes_per_frame_dev
                if leftover > 0:
                    raw = bytes(audio_buffer[:leftover])
                    del audio_buffer[:leftover]
                    arr = np.frombuffer(raw, dtype=np.int16)
                    ch = min(info["max_input_channels"], 2)
                    if ch > 1:
                        arr = arr.reshape(-1, ch)
                    mono_int16 = _downmix_to_mono_int16(arr)
                    mono_16k_float = _resample_linear(mono_int16, device_sr, TARGET_SR)
                    chunk_start_sec = session.total_target_samples / TARGET_SR
                    session.total_target_samples += mono_16k_float.size
                    event = _transcribe_chunk(
                        session=session,
                        mono_16k_float=mono_16k_float,
                        chunk_index=session.chunk_index,
                        final_chunk=True,
                        chunk_start_sec=chunk_start_sec,
                    )
                    if event:
                        session.queue.put(event)
                    session.chunk_index += 1
            else:
                # Emit final empty event
                t = session.total_target_samples / TARGET_SR
                final_event = {
                    "session_id": session.session_id,
                    "chunk_index": session.chunk_index,
                    "start_sec": t,
                    "end_sec": t,
                    "text": "",
                    "words": [],
                    "final": True,
                }
                session.queue.put(final_event)

        logger.info("Session %s finished audio capture", session.session_id)
    except Exception as exc:
        logger.exception("Session %s encountered an error: %s", session.session_id, exc)
        error_event = {
            "session_id": session.session_id,
            "chunk_index": session.chunk_index,
            "start_sec": session.total_target_samples / TARGET_SR,
            "end_sec": session.total_target_samples / TARGET_SR,
            "text": "",
            "words": [],
            "final": True,
        }
        session.queue.put(error_event)
    finally:
        session.queue.put(None)
        session.active = False        # mark inactive so /session/start can run again
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
        session.thread.join(timeout=5)
    session.finished_event.wait(timeout=5)


app = FastAPI(title="ASR Service", version="1.2.0")


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
