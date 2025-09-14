# rover-learn/services/asr/server.py
from __future__ import annotations

import base64
import os
import time
from typing import Optional
from datetime import datetime

import webrtcvad

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

_vad = webrtcvad.Vad(2)
_locked_lang: Optional[str] = os.getenv("ASR_FORCE_LANG")
_curr_lang: str = _locked_lang or "en"
_para_text: str = ""
_para_id: Optional[str] = None
_para_start_t: float = 0.0
_last_voiced_t: float = 0.0
_para_counter: int = 0


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


def _voiced_ratio(pcm16: np.ndarray, sr: int) -> float:
    if pcm16.size < 320:
        return 0.0
    frame_len = int(sr / 50)  # 20 ms
    pcm_bytes = pcm16.tobytes()
    voiced = 0
    total = 0
    for i in range(0, len(pcm_bytes), frame_len * 2):
        frame = pcm_bytes[i : i + frame_len * 2]
        if len(frame) < frame_len * 2:
            break
        if _vad.is_speech(frame, sr):
            voiced += 1
        total += 1
    return voiced / total if total else 0.0


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

    # compute chunk length and base offset
    chunk_len_s = len(pcm16) / float(sr)
    base_offset = _offset_seconds_for_idx(payload.idx, payload.chunk_ms, chunk_len_s)

    # VAD gate
    ratio = _voiced_ratio(pcm16, sr)
    out = []
    global _para_text, _para_id, _para_start_t, _last_voiced_t, _para_counter, _locked_lang, _curr_lang
    if ratio < 0.1:
        # possible silence: check endpointing
        if _para_text and (base_offset - _last_voiced_t) > 0.6:
            seg = {
                "kind": "final",
                "paraId": _para_id,
                "idx": int(payload.idx),
                "tStart": _para_start_t,
                "tEnd": _last_voiced_t,
                "lang": _curr_lang,
                "speaker": "Speaker 1",
                "textSrc": _para_text.strip(),
                "partial": False,
                "confidence": 0.9,
            }
            out.append(seg)
            _para_text = ""
            _para_id = None
        _tail_pcm = pcm16[-_TAIL_S:] if pcm16.size > _TAIL_S else pcm16.copy()
        return {"segments": out}

    _last_voiced_t = base_offset + chunk_len_s

    # append overlap tail and prepare audio
    pcm16_full = np.concatenate([_tail_pcm, pcm16]) if _tail_pcm.size else pcm16
    audio_f32 = _to_float32(pcm16_full)

    lang_arg = _locked_lang if _locked_lang else None

    segments, info = _MODEL.transcribe(
        audio_f32,
        language=lang_arg,
        vad_filter=False,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False,
        initial_prompt=None,
        word_timestamps=False,
        no_speech_threshold=0.5,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    lang = getattr(info, "language", None) or (_locked_lang or "en")
    _curr_lang = lang
    if not _locked_lang:
        _locked_lang = lang

    tail_shift = (_TAIL_S / float(sr)) if _tail_pcm.size else 0.0
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue
        seg_start = float(s.start or 0.0)
        if not _para_id:
            _para_counter += 1
            _para_start_t = max(0.0, base_offset + max(0.0, seg_start - tail_shift))
            _para_id = f"{datetime.utcnow().isoformat()}#{_para_counter}"
            _para_text = ""
        _para_text = (_para_text + " " + text).strip()

    if _para_text:
        out.append(
            {
                "kind": "partial",
                "paraId": _para_id,
                "idx": int(payload.idx),
                "lang": lang,
                "textSrcPartial": _para_text,
                "partial": True,
            }
        )

    _tail_pcm = pcm16[-_TAIL_S:] if pcm16.size > _TAIL_S else pcm16.copy()

    return {"segments": out}
