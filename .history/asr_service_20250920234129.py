"""ASR microservice using FastAPI and faster-whisper (Windows-optimized, GPU, 0.75s chunks)."""
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
# Config (tuned for you)
# =======================
TARGET_SR = 16000            # Whisper prefers ~16 kHz when passing numpy audio
SAMPLE_WIDTH = 2             # 16-bit capture
CHUNK_SEC = 0.75             # ~0.75 s latency

MODEL_NAME   = os.getenv("WHISPER_MODEL", "medium")
DEVICE       = "cuda"        # GPU only (your requirement)
COMPUTE_TYPE = "int8_float16"
BEAM_SIZE    = 5
VAD_FILTER   = False         # keep off (no onnxruntime), instant UX
WORD_TIMESTAMPS = True

# Device picking
ASR_INPUT_DEVICE_ENV = os.getenv("ASR_INPUT_DEVICE")  # index or substring
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
    total_target_samples: int = 0  # samples at 16kHz timeline
    active: bool = True            # set False when the capture thread exits


sessions: Dict[str, SessionState] = {}
sessions_lock = threading.Lock()
model_lock = threading.Lock()
MODEL: Optional[WhisperModel] = None
CURRENT_DEVICE = DEVICE
CURRENT_COMPUTE_TYPE = COMPUTE_TYPE


class StopSessionRequest(BaseModel):
    session_id: str


def init_model() -> None:
    """Load Whisper on GPU (no CPU fallback)."""
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


def _pick_input_device() -> Tuple[Optional[int], str, int, int]:
    """
    Returns (device_index, label, max_input_channels, default_samplerate).
    Prefers WASAPI mic; honors ASR_INPUT_DEVICE env (index or substring).
    """
    try:
        hostapis = sd.query_hostapis()
        ha_names = {i: ha.get("name", "") for i, ha in enumerate(hostapis)}
    except Exception as e:
        logger.warning("Unable to query hostapis: %s", e)
        ha_names = {}

    devices = sd.query_devices()
    default_in, _ = sd.default.device
    chosen_idx: Optional[int] = None
    chosen_label = ""
    chosen_ch = 0
    chosen_sr = 48000

    # env override
    if ASR_INPUT_DEVICE_ENV:
        val = ASR_INPUT_DEVICE_ENV.strip()
        # try as index
        try:
            idx = int(val)
            info = sd.query_devices(idx)
            if info["max_input_channels"] > 0:
                chosen_idx = idx
                chosen_ch = int(info["max_input_channels"])
                chosen_sr = int(round(info.get("default_samplerate") or 48000))
                chosen_label = f"[env index {idx}] {info['name']} ({ha_names.get(info['hostapi'], '')})"
                return chosen_idx, chosen_label, chosen_ch, chosen_sr
        except Exception:
            pass
        # try name match
        val_l = val.lower()
        for i, info in enumerate(devices):
            if info.get("max_input_channels", 0) <= 0:
                continue
            if val_l in str(info.get("name", "")).lower():
                chosen_idx = i
                chosen_ch = int(info["max_input_channels"])
                chosen_sr = int(round(info.get("default_samplerate") or 48000))
                chosen_label = f"[env match '{val}'] {info['name']} ({ha_names.get(info['hostapi'], '')})"
                return chosen_idx, chosen_label, chosen_ch, chosen_sr
        logger.warning("ASR_INPUT_DEVICE='%s' not found/usable; falling back to auto.", ASR_INPUT_DEVICE_ENV)

    # WASAPI + mic hint
    for i, info in enumerate(devices):
        if info.get("max_input_channels", 0) <= 0:
            continue
        host = ha_names.get(info.get("hostapi", -1), "")
        if PREFERRED_HOSTAPI.lower() in host.lower():
            name_l = str(info.get("name", "")).lower()
            if any(k in name_l for k in MIC_NAME_HINTS):
                chosen_idx = i
                chosen_ch = int(info["max_input_channels"])
                chosen_sr = int(round(info.get("default_samplerate") or 48000))
                chosen_label = f"[auto WASAPI+mic] {info['name']} ({host})"
                return chosen_idx, chosen_label, chosen_ch, chosen_sr

    # any WASAPI input
    for i, info in enumerate(devices):
        if info.get("max_input_channels", 0) <= 0:
            continue
        host = ha_names.get(info.get("hostapi", -1), "")
        if PREFERRED_HOSTAPI.lower() in host.lower():
            chosen_idx = i
            chosen_ch = int(info["max_input_channels"])
            chosen_sr = int(round(info.get("default_samplerate") or 48000))
            chosen_label = f"[auto WASAPI] {info['name']} ({host})"
            return chosen_idx, chosen_label, chosen_ch, chosen_sr

    # default input
    try:
        info = sd.query_devices(default_in)
        if info.get("max_input_channels", 0) > 0:
            chosen_idx = default_in
            chosen_ch = int(info["max_input_channels"])
            chosen_sr = int(round(info.get("default_samplerate") or 48000))
            chosen_label = f"[default] {info['name']} ({ha_names.get(info['hostapi'], '')})"
            return chosen_idx, chosen_label, chosen_ch, chosen_sr
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
) -> dict:
    """
    Always emits an event (even if text is empty). That guarantees the UI sees activity
    and the session can cleanly finalize.
    """
    with model_lock:
        if MODEL is None:
            init_model()
        model = MODEL
    assert model is not None

    text_parts = []
    words_payload = []
    rel_start = 0.0
    rel_end = mono_16k_float.size / TARGET_SR

    # Only run the model if there is non-trivial energy or it's final.
    # Lower gate to be very permissive so we don't drop speech.
    if mono_16k_float.size > 0:
        rms = float(np.sqrt(np.mean(np.square(mono_16k_float))) + 1e-12)
    else:
        rms = 0.0

    if rms > 1e-7 or final_chunk:
        segments, info = model.transcribe(
            audio=mono_16k_float,
            language=None,              # auto-detect
            beam_size=BEAM_SIZE,
            vad_filter=VAD_FILTER,
            word_timestamps=WORD_TIMESTAMPS,
        )

        got_ts = False
        for segment in segments:
            cleaned = segment.text.strip()
            if cleaned:
                text_parts.append(cleaned)
            if segment.words:
                for w in segment.words:
                    s = None if w.start is None else round(chunk_start_sec + w.start, 2)
                    e = None if w.end   is None else round(chunk_start_sec + w.end,   2)
                    words_payload.append({"start": s, "end": e, "word": w.word})
            if (segment.start is not None) and not got_ts:
                rel_start = float(segment.start)
                got_ts = True
            if segment.end is not None:
                rel_end = float(segment.end)

    event = {
        "session_id": session.session_id,
        "chunk_index": chunk_index,
        "start_sec": round(chunk_start_sec + rel_start, 2),
        "end_sec": round(chunk_start_sec + rel_end, 2),
        "text": " ".join(text_parts).strip(),
        "words": words_payload,
        "final": final_chunk,
    }
    return event


def _capture_and_transcribe(session: SessionState) -> None:
    dev_idx, dev_label, max_in_ch, device_sr = _pick_input_device()
    if dev_idx is None or max_in_ch <= 0:
        logger.error("No usable input device found. Check Windows Sound settings and privacy.")
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

    device_sr = int(max(8000, min(192000, round(device_sr))))
    in_channels = min(max_in_ch, 2)  # up to stereo
    logger.info("Using input device #%s %s @ %d Hz", dev_idx, dev_label, device_sr)

    chunk_samples_dev = int(device_sr * CHUNK_SEC)
    read_frames = int(device_sr * 0.5)
    bytes_per_frame_dev = SAMPLE_WIDTH * in_channels
    audio_buffer = bytearray()

    try:
        with sd.InputStream(
            device=dev_idx,
            samplerate=device_sr,
            channels=in_channels,
            dtype="int16",
            blocksize=0,
            latency="low",
        ) as stream:
            logger.info("Session %s started audio capture on %s/%s", session.session_id, CURRENT_DEVICE, CURRENT_COMPUTE_TYPE)
            while not session.stop_event.is_set():
                frames, overflowed = stream.read(read_frames)
                if overflowed:
                    logger.warning("Audio input overflow detected for session %s", session.session_id)
                audio_buffer.extend(frames.tobytes())

                while len(audio_buffer) >= chunk_samples_dev * bytes_per_frame_dev:
                    raw = bytes(audio_buffer[: chunk_samples_dev * bytes_per_frame_dev])
                    del audio_buffer[: chunk_samples_dev * bytes_per_frame_dev]

                    arr = np.frombuffer(raw, dtype=np.int16)
                    if in_channels > 1:
                        arr = arr.reshape(-1, in_channels)
                    mono_int16 = _downmix_to_mono_int16(arr)
                    mono_16k = _resample_linear(mono_int16, device_sr, TARGET_SR)

                    chunk_start_sec = session.total_target_samples / TARGET_SR
                    session.total_target_samples += mono_16k.size

                    event = _transcribe_chunk(
                        session=session,
                        mono_16k_float=mono_16k,
                        chunk_index=session.chunk_index,
                        final_chunk=False,
                        chunk_start_sec=chunk_start_sec,
                    )
                    session.queue.put(event)  # ALWAYS enqueue (text may be "")
                    session.chunk_index += 1

            # flush remaining buffer as final
            leftover = len(audio_buffer) // bytes_per_frame_dev * bytes_per_frame_dev
            if leftover > 0:
                raw = bytes(audio_buffer[:leftover])
                del audio_buffer[:leftover]
                arr = np.frombuffer(raw, dtype=np.int16)
                if in_channels > 1:
                    arr = arr.reshape(-1, in_channels)
                mono_int16 = _downmix_to_mono_int16(arr)
                mono_16k = _resample_linear(mono_int16, device_sr, TARGET_SR)
            else:
                mono_16k = np.array([], dtype=np.float32)

            chunk_start_sec = session.total_target_samples / TARGET_SR
            session.total_target_samples += mono_16k.size
            event = _transcribe_chunk(
                session=session,
                mono_16k_float=mono_16k,
                chunk_index=session.chunk_index,
                final_chunk=True,
                chunk_start_sec=chunk_start_sec,
            )
            session.queue.put(event)

        logger.info("Session %s finished audio capture", session.session_id)
    except Exception as exc:
        logger.exception("Session %s encountered an error: %s", session.session_id, exc)
        # still signal end
        t = session.total_target_samples / TARGET_SR
        session.queue.put({
            "session_id": session.session_id,
            "chunk_index": session.chunk_index,
            "start_sec": t,
            "end_sec": t,
            "text": "",
            "words": [],
            "final": True,
        })
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
    logger.info("Stopping session %s on request", session_id)
    session.stop_event.set()
    if session.thread and session.thread.is_alive():
        session.thread.join(timeout=5)
    session.finished_event.wait(timeout=5)
    # safe to cleanup after the thread has finished
    with sessions_lock:
        sessions.pop(session_id, None)
    logger.info("Session %s fully stopped and cleaned up", session_id)


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
        # IMPORTANT: do NOT clean up session here. The Stop endpoint owns cleanup,
        # so /session/stop can always find the session and set stop_event.
        while True:
            if await request.is_disconnected():
                logger.info("Client disconnected from session %s stream", session_id)
                break
            item = await loop.run_in_executor(None, session_queue.get)
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


atexit.register(close_model)
