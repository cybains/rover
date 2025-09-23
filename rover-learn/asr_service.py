import asyncio
import json
import logging
import os
import threading
import uuid
import wave
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, List, Optional

import numpy as np
import sounddevice as sd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from faster_whisper import WhisperModel
from pydantic import BaseModel

TARGET_SR = 16_000
CHUNK_SEC_CAPTURE = 0.5
CHUNK_SEC_FILE = 5.0
TRANSCRIBE_WINDOW_SEC = 8.0
TRANSCRIBE_OVERLAP_SEC = 0.8
MODEL_NAME = os.getenv("WHISPER_MODEL", "medium")
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"
BEAM_SIZE = 5
USE_VAD = False
WORD_TS = False
FILE_SAMPLES = int(CHUNK_SEC_FILE * TARGET_SR)

DATA_ROOT = Path(__file__).resolve().parent / ".data"
SESSIONS_ROOT = DATA_ROOT / "sessions"

logger = logging.getLogger("rover.asr")
logging.basicConfig(level=logging.INFO)


def ensure_data_dirs() -> None:
    DATA_ROOT.mkdir(exist_ok=True)
    SESSIONS_ROOT.mkdir(exist_ok=True)


class StopRequest(BaseModel):
    session_id: str


@dataclass
class SessionState:
    session_id: str
    source: str
    loop: asyncio.AbstractEventLoop
    audio_dir: Path
    event_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    stop_event: threading.Event = field(default_factory=threading.Event)
    data_event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    capture_thread: Optional[threading.Thread] = None
    transcribe_thread: Optional[threading.Thread] = None
    running: bool = False
    total_samples: int = 0
    processed_cursor: int = 0
    history_start: int = 0
    history_samples: int = 0
    history: Deque[np.ndarray] = field(default_factory=deque)
    file_chunks: List[np.ndarray] = field(default_factory=list)
    file_samples: int = 0
    part_index: int = 0
    chunk_index: int = 0
    last_final_end: float = 0.0
    error: Optional[str] = None

    def append_audio(self, data: np.ndarray) -> None:
        with self.lock:
            self.history.append(data)
            self.history_samples += len(data)
            self.total_samples += len(data)
            self.file_chunks.append(data)
            self.file_samples += len(data)
            max_history = int((TRANSCRIBE_WINDOW_SEC + TRANSCRIBE_OVERLAP_SEC) * TARGET_SR * 2)
            while self.history_samples > max_history and self.history:
                removed = self.history.popleft()
                self.history_samples -= len(removed)
                self.history_start += len(removed)

    def consume_file_buffer(self, force: bool = False) -> Optional[np.ndarray]:
        with self.lock:
            if not self.file_chunks:
                return None
            if not force and self.file_samples < FILE_SAMPLES:
                return None
            data = (
                np.concatenate(self.file_chunks)
                if len(self.file_chunks) > 1
                else self.file_chunks[0]
            )
            self.file_chunks.clear()
            self.file_samples = 0
            return data

    def get_audio_span(self, start_sample: int, end_sample: int) -> np.ndarray:
        with self.lock:
            if not self.history:
                return np.empty(0, dtype=np.float32)
            start_sample = max(start_sample, self.history_start)
            end_sample = max(end_sample, start_sample)
            offset = start_sample - self.history_start
            length = end_sample - start_sample
            if length <= 0:
                return np.empty(0, dtype=np.float32)
            arrays = list(self.history)
        joined = (
            np.concatenate(arrays)
            if len(arrays) > 1
            else arrays[0].copy()
        )
        return joined[offset : offset + length]

    def queue_event(self, payload: dict) -> None:
        payload.setdefault("session_id", self.session_id)
        asyncio.run_coroutine_threadsafe(
            self.event_queue.put(payload), self.loop
        )

    def close_queue(self) -> None:
        asyncio.run_coroutine_threadsafe(self.event_queue.put(None), self.loop)


def normalise_audio(data: np.ndarray, samplerate: int, channels: int) -> np.ndarray:
    if data.ndim == 2 and data.shape[1] > 1:
        data = data.mean(axis=1)
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.float32:
        data = data.astype(np.float32)
    else:
        data = data.astype(np.float32)
    if samplerate != TARGET_SR and len(data) > 0:
        duration = len(data) / samplerate
        target_len = int(duration * TARGET_SR)
        if target_len <= 0:
            return np.empty(0, dtype=np.float32)
        orig_times = np.linspace(0.0, duration, num=len(data), endpoint=False)
        target_times = np.linspace(0.0, duration, num=target_len, endpoint=False)
        data = np.interp(target_times, orig_times, data).astype(np.float32)
    return data


def write_wav_part(session: SessionState, audio: np.ndarray) -> None:
    if audio.size == 0:
        return
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    part_path = session.audio_dir / f"part_{session.part_index:04d}.wav"
    session.part_index += 1
    with wave.open(part_path.as_posix(), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm.tobytes())
    logger.debug("Session %s wrote %s", session.session_id, part_path.name)


def select_device(source: str) -> dict:
    try:
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
    except Exception as exc:  # pragma: no cover - environment specific
        logger.warning("Could not query audio devices: %s", exc)
        return {
            "device": None,
            "samplerate": TARGET_SR,
            "channels": 1,
            "extra_settings": None,
        }

    wasapi_id = None
    for idx, host in enumerate(hostapis):
        if "wasapi" in host["name"].lower():
            wasapi_id = idx
            break

    def pick(predicate, prefer_loopback=False, channels_key="max_input_channels"):
        for idx, dev in enumerate(devices):
            if predicate(idx, dev):
                extra = None
                if prefer_loopback and hasattr(sd, "WasapiSettings"):
                    extra = sd.WasapiSettings(loopback=True)
                channels = int(dev.get(channels_key, 0) or 1)
                return {
                    "device": idx,
                    "samplerate": int(dev.get("default_samplerate") or TARGET_SR),
                    "channels": max(1, channels),
                    "extra_settings": extra,
                }
        return None

    choice = None
    if source == "system":
        if wasapi_id is not None:
            choice = pick(
                lambda idx, dev: dev["hostapi"] == wasapi_id
                and dev.get("max_output_channels", 0) > 0,
                prefer_loopback=True,
                channels_key="max_output_channels",
            )
    elif source == "mic":
        choice = pick(lambda idx, dev: dev.get("max_input_channels", 0) > 0)
    elif source == "auto":
        choice = pick(lambda idx, dev: dev.get("max_input_channels", 0) > 0)
        if not choice and wasapi_id is not None:
            choice = pick(
                lambda idx, dev: dev["hostapi"] == wasapi_id
                and dev.get("max_output_channels", 0) > 0,
                prefer_loopback=True,
                channels_key="max_output_channels",
            )

    if not choice:
        defaults = sd.default.device
        device = defaults[0] if isinstance(defaults, (list, tuple)) else defaults
        samplerate = (
            int(sd.query_devices(device)["default_samplerate"])
            if device is not None
            else TARGET_SR
        )
        channels = 1
        choice = {
            "device": device,
            "samplerate": samplerate or TARGET_SR,
            "channels": channels,
            "extra_settings": None,
        }
    return choice


def capture_worker(session: SessionState) -> None:
    logger.info("Starting capture for session %s (%s)", session.session_id, session.source)
    settings = select_device(session.source)
    samplerate = settings.get("samplerate", TARGET_SR) or TARGET_SR
    channels = max(1, settings.get("channels", 1))
    device_label = "default"
    try:
        device_id = settings.get("device")
        if device_id is not None:
            info = sd.query_devices(device_id)
            device_label = info.get("name", str(device_id))
    except Exception:  # pragma: no cover - device lookup is best effort
        pass
    logger.info(
        "Session %s using device %s @ %s Hz (%s ch)",
        session.session_id,
        device_label,
        samplerate,
        channels,
    )
    blocksize = max(1, int(samplerate * CHUNK_SEC_CAPTURE))

    try:
        with sd.InputStream(
            samplerate=samplerate,
            blocksize=blocksize,
            device=settings.get("device"),
            dtype="int16",
            channels=channels,
            extra_settings=settings.get("extra_settings"),
        ) as stream:
            session.running = True
            while not session.stop_event.is_set():
                frames, overflowed = stream.read(blocksize)
                if overflowed:
                    logger.warning("Audio overflow on session %s", session.session_id)
                audio = normalise_audio(frames, samplerate, channels)
                if audio.size == 0:
                    continue
                session.append_audio(audio)
                session.data_event.set()
                file_audio = session.consume_file_buffer()
                if file_audio is not None:
                    write_wav_part(session, file_audio)
    except Exception as exc:
        session.error = str(exc)
        logger.error("Capture failed for session %s: %s", session.session_id, exc)
    finally:
        session.running = False
        remainder = session.consume_file_buffer(force=True)
        if remainder is not None:
            write_wav_part(session, remainder)
        session.data_event.set()
        logger.info("Capture ended for session %s", session.session_id)


def transcribe_worker(session: SessionState, model: WhisperModel) -> None:
    logger.info("Starting transcription for session %s", session.session_id)
    overlap_samples = int(TRANSCRIBE_OVERLAP_SEC * TARGET_SR)
    min_step = int(CHUNK_SEC_CAPTURE * TARGET_SR)

    while True:
        session.data_event.wait(timeout=0.5)
        session.data_event.clear()

        while True:
            if session.stop_event.is_set() and not session.running:
                break
            total = session.total_samples
            processed = session.processed_cursor
            if total - processed < min_step and not session.stop_event.is_set():
                break
            start_sample = max(0, processed - overlap_samples)
            end_sample = total
            audio = session.get_audio_span(start_sample, end_sample)
            if audio.size == 0:
                break
            try:
                segments, _ = model.transcribe(
                    audio,
                    language=None,
                    beam_size=BEAM_SIZE,
                    vad_filter=USE_VAD,
                    word_timestamps=WORD_TS,
                )
            except Exception as exc:
                logger.error("Transcription error on session %s: %s", session.session_id, exc)
                session.queue_event(
                    {
                        "chunk_index": session.chunk_index,
                        "start_sec": start_sample / TARGET_SR,
                        "end_sec": end_sample / TARGET_SR,
                        "text": "",
                        "final": True,
                    }
                )
                session.chunk_index += 1
                session.processed_cursor = end_sample
                break

            new_segment_emitted = False
            for seg in segments:
                abs_start = start_sample / TARGET_SR + seg.start
                abs_end = start_sample / TARGET_SR + seg.end
                if abs_end <= session.last_final_end + 1e-3:
                    continue
                text = seg.text.strip()
                payload = {
                    "chunk_index": session.chunk_index,
                    "start_sec": abs_start,
                    "end_sec": abs_end,
                    "text": text,
                    "final": False,
                }
                session.queue_event(payload.copy())
                payload["final"] = True
                session.queue_event(payload)
                session.chunk_index += 1
                session.last_final_end = abs_end
                new_segment_emitted = True

            if not new_segment_emitted and end_sample > processed:
                payload = {
                    "chunk_index": session.chunk_index,
                    "start_sec": start_sample / TARGET_SR,
                    "end_sec": end_sample / TARGET_SR,
                    "text": "",
                    "final": False,
                }
                session.queue_event(payload.copy())
                payload["final"] = True
                session.queue_event(payload)
                session.chunk_index += 1

            session.processed_cursor = end_sample

            if session.stop_event.is_set() and not session.running:
                break

        if session.stop_event.is_set() and not session.running:
            break

    session.close_queue()
    logger.info("Transcription ended for session %s", session.session_id)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    ensure_data_dirs()
    loop = asyncio.get_running_loop()
    model = WhisperModel(
        MODEL_NAME,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
        download_root=str(DATA_ROOT / "models"),
    )
    app.state.model = model
    app.state.loop = loop
    app.state.sessions = {}
    logger.info("Whisper model %s loaded on %s (%s)", MODEL_NAME, DEVICE, COMPUTE_TYPE)


@app.on_event("shutdown")
async def shutdown_event() -> None:
    sessions = list(app.state.sessions.values())
    for session in sessions:
        session.stop_event.set()
        session.data_event.set()
        if session.capture_thread:
            session.capture_thread.join(timeout=2.0)
        if session.transcribe_thread:
            session.transcribe_thread.join(timeout=2.0)
        session.close_queue()
    model: WhisperModel = app.state.model
    if model is not None:
        model.close()
    logger.info("ASR service shutdown complete")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.post("/session/start")
async def session_start(source: str = Query("auto", pattern="^(mic|system|auto)$")) -> JSONResponse:
    loop: asyncio.AbstractEventLoop = app.state.loop
    model: WhisperModel = app.state.model
    if model is None:
        raise HTTPException(status_code=500, detail="Model not initialised")

    session_id = str(uuid.uuid4())
    audio_dir = SESSIONS_ROOT / session_id / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    session = SessionState(session_id=session_id, source=source, loop=loop, audio_dir=audio_dir)
    capture = threading.Thread(target=capture_worker, args=(session,), daemon=True)
    transcribe = threading.Thread(
        target=transcribe_worker, args=(session, model), daemon=True
    )
    session.capture_thread = capture
    session.transcribe_thread = transcribe
    app.state.sessions[session_id] = session

    capture.start()
    transcribe.start()

    logger.info("Session %s started (%s)", session_id, source)
    return JSONResponse({"session_id": session_id, "source": source})


@app.post("/session/stop")
async def session_stop(payload: StopRequest) -> JSONResponse:
    session_id = payload.session_id
    session: SessionState = app.state.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.stop_event.set()
    session.data_event.set()
    if session.capture_thread:
        session.capture_thread.join(timeout=2.0)
    if session.transcribe_thread:
        session.transcribe_thread.join(timeout=2.0)
    session.close_queue()
    app.state.sessions.pop(session_id, None)
    logger.info("Session %s stopped", session_id)
    return JSONResponse({"status": "stopped", "session_id": session_id})


@app.get("/session/{session_id}/stream")
async def session_stream(session_id: str):
    session: SessionState = app.state.sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    async def event_generator():
        while True:
            event = await session.event_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
