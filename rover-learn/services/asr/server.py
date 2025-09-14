# rover-learn/services/asr/server.py
from __future__ import annotations

import base64
import os
import time
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel

app = FastAPI()

# ---------- Model & Streaming Context ----------

_MODEL: Optional[WhisperModel] = None

# single-stream context (one live user at a time)
_SR = 16000                   # expected sample rate
_TAIL_S = int(0.2 * _SR)      # 200 ms overlap tail (shorter to avoid repeats)
_tail_pcm = np.zeros(0, dtype=np.int16)
_prev_text: str = ""          # (unused when conditioning disabled, kept for future)


def _init_model():
    """
    Model/device are set from env to allow CUDA:
      ASR_MODEL      (default: "small")
      ASR_DEVICE     (default: "cpu")      -> set "cuda" to use your GTX 1650
      ASR_COMPUTE    (default: "int8")     -> set "float16" on CUDA
      ASR_FORCE_LANG (optional: "de")      -> skip lang detection
    Returns (model, model_id, device, compute_type, load_ms)
    """
    model_id = os.getenv("ASR_MODEL", "small")
    device = os.getenv("ASR_DEVICE", "cpu")
    compute_type = os.getenv("ASR_COMPUTE", "int8")

    t0 = time.time()
    model = WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
        cpu_threads=max(4, os.cpu_count() or 8),
    )
    load_ms = (time.time() - t0) * 1000.0
    return model, model_id, device, compute_type, load_ms


# ---------- DTOs ----------

class TranscribeIn(BaseModel):
    text: str
    idx: int


class ChunkIn(BaseModel):
    audio_b16: str
    sample_rate: int
    idx: int
    chunk_ms: int | None = None   # <- used to align timestamps exactly


# ---------- Helpers ----------

def _to_float32(pcm16: np.ndarray) -> np.ndarray:
    # int16 -> float32 in [-1, 1]
    return (pcm16.astype(np.float32)) / 32768.0


def _offset_seconds_for_idx(idx: int, chunk_ms: int | None, fallback_chunk_len_s: float) -> float:
    # place this chunk on a global timeline by index
    if chunk_ms and chunk_ms > 0:
        return max(0.0, float(idx) * (float(chunk_ms) / 1000.0))
    return max(0.0, float(idx) * float(fallback_chunk_len_s))


# ---------- FastAPI Lifecycle ----------

@app.on_event("startup")
def _load_model():
    global _MODEL
    _MODEL, model_id, device, compute_type, load_ms = _init_model()
    print(
        f"[ASR] model={model_id} loaded in {load_ms:.0f} ms (device={device} compute={compute_type})"
    )


@app.get("/health")
def health():
    return {"service": "asr", "status": "ok"}


# ---------- Simple text-based path (compat) ----------

@app.post("/transcribe")
def transcribe(payload: TranscribeIn):
    # Kept for compatibility/testing
    idx = payload.idx
    return {
        "idx": idx,
        "tStart": idx * 1.0,
        "tEnd": idx * 1.0 + 0.9,
        "lang": "de",
        "speaker": "Speaker 1",
        "textSrc": payload.text,
        "partial": False,
        "confidence": 0.95,
    }


# ---------- Chunk streaming path (realtime) ----------

@app.post("/transcribe_chunk")
def transcribe_chunk(payload: ChunkIn):
    """
    Accepts base64 PCM16 mono @ sample_rate.
    Uses short overlap + exact chunk-based timeline. Decoding is tuned
    for low-latency, no cross-chunk conditioning (prevents repeats).
    """
    global _MODEL, _tail_pcm, _prev_text

    if _MODEL is None:
        return {"segments": []}

    # validate / decode
    try:
        sr = int(payload.sample_rate)
        if sr <= 0:
            return {"segments": []}
    except Exception:
        return {"segments": []}

    try:
        audio_bytes = base64.b64decode(payload.audio_b16)
    except Exception:
        return {"segments": []}

    pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
    if pcm16.ndim != 1:
        pcm16 = pcm16.reshape(-1)

    if len(pcm16) < int(0.15 * sr):
        return {"segments": []}
    if np.max(np.abs(pcm16)) < 200:
        return {"segments": []}

    # append overlap tail
    pcm16_full = np.concatenate([_tail_pcm, pcm16]) if _tail_pcm.size else pcm16

    # compute chunk length (based on current chunk) and a global offset
    chunk_len_s = len(pcm16) / float(sr)
    base_offset = _offset_seconds_for_idx(payload.idx, payload.chunk_ms, chunk_len_s)

    # prepare audio
    audio_f32 = _to_float32(pcm16_full)

    # language: optionally force German to skip detection
    force_lang = os.getenv("ASR_FORCE_LANG")
    lang_arg = force_lang if force_lang else None

    # low-latency decode settings (beam_size=1, no conditioning)
    segments, info = _MODEL.transcribe(
        audio_f32,
        language=lang_arg,                 # None => autodetect; "de" recommended for speed
        vad_filter=False,                  # avoid onnxruntime dependency
        beam_size=1,                       # faster on 1650
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,  # <- no cross-chunk repeats
        initial_prompt=None,
        word_timestamps=False,
        no_speech_threshold=0.5,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    lang = (getattr(info, "language", None) or (force_lang or "en"))

    out = []
    tail_shift = (_TAIL_S / float(sr)) if _tail_pcm.size else 0.0

    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue

        # segment times are relative to pcm16_full (starts with tail)
        seg_start = float(s.start or 0.0)
        seg_end = float(s.end or 0.0)

        # subtract tail, then add base offset
        t_start = max(0.0, base_offset + max(0.0, seg_start - tail_shift))
        t_end = max(t_start, base_offset + max(0.0, seg_end - tail_shift))

        out.append(
            {
                "idx": int(payload.idx),
                "tStart": t_start,
                "tEnd": t_end,
                "lang": lang,
                "speaker": "Speaker 1",
                "textSrc": text,
                "partial": False,
                "confidence": 0.9,
            }
        )

    # update overlap tail
    _tail_pcm = pcm16[-_TAIL_S:] if pcm16.size > _TAIL_S else pcm16.copy()

    return {"segments": out}
