# rover-learn/services/asr/server.py
from __future__ import annotations

import base64
import os
from typing import Optional

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel

app = FastAPI()

# ---------- Model & Streaming Context ----------

_MODEL: Optional[WhisperModel] = None

# single-stream context (one live user at a time)
_SR = 16000                  # expected sample rate
_TAIL_S = int(0.3 * _SR)     # 300 ms overlap tail
_tail_pcm = np.zeros(0, dtype=np.int16)
_prev_text: str = ""         # rolling prompt memory (stabilizes decoding)


def _init_model() -> WhisperModel:
    """
    CPU-first preset. Override via env if you later get GPU wheels working.
      ASR_MODEL      (default: "small")
      ASR_DEVICE     (default: "cpu")
      ASR_COMPUTE    (default: "int8" for cpu)
    """
    model_id = os.getenv("ASR_MODEL", "small")     # good multilingual CPU balance
    device = os.getenv("ASR_DEVICE", "cpu")        # stick to cpu in your env
    compute_type = os.getenv("ASR_COMPUTE", "int8")

    return WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
        cpu_threads=max(4, os.cpu_count() or 8),
    )


# ---------- DTOs ----------

GERMAN_TOKENS = {"der", "die", "und"}


class TranscribeIn(BaseModel):
    text: str
    idx: int


class ChunkIn(BaseModel):
    audio_b16: str
    sample_rate: int
    idx: int
    # (optional future) sessionId: str | None = None


# ---------- Helpers ----------

def _detect_lang_text(text: str) -> str:
    t = text.lower()
    if any(tok in t.split() for tok in GERMAN_TOKENS) or any(ch in t for ch in "äöüß"):
        return "de"
    return "en"


def _to_float32(pcm16: np.ndarray) -> np.ndarray:
    # int16 -> float32 in [-1, 1]
    return (pcm16.astype(np.float32)) / 32768.0


def _offset_seconds_for_idx(idx: int, chunk_len_s: float) -> float:
    # place this chunk on a global timeline by index
    return max(0.0, float(idx) * float(chunk_len_s))


# ---------- FastAPI Lifecycle ----------

@app.on_event("startup")
def _load_model():
    global _MODEL
    _MODEL = _init_model()


@app.get("/health")
def health():
    return {"service": "asr", "status": "ok"}


# ---------- Simple text-based path (compat) ----------

@app.post("/transcribe")
def transcribe(payload: TranscribeIn):
    lang = _detect_lang_text(payload.text)
    idx = payload.idx
    return {
        "idx": idx,
        "tStart": idx * 1.0,
        "tEnd": idx * 1.0 + 0.9,
        "lang": lang,
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
    Uses 300ms overlap + rolling prompt for stability, and returns segments with
    proper global timestamps based on chunk index and actual chunk duration.
    """
    global _MODEL, _tail_pcm, _prev_text

    if _MODEL is None:
        return {"segments": []}

    # basic validation
    try:
        sr = int(payload.sample_rate)
    except Exception:
        return {"segments": []}
    if sr <= 0:
        return {"segments": []}

    # decode to PCM16 mono
    try:
        audio_bytes = base64.b64decode(payload.audio_b16)
    except Exception:
        return {"segments": []}
    pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)
    if pcm16.ndim != 1:
        pcm16 = pcm16.reshape(-1)

    # append overlap tail
    pcm16_full = np.concatenate([_tail_pcm, pcm16]) if _tail_pcm.size else pcm16

    # compute chunk length (based on *current* chunk only, not including tail)
    chunk_len_s = len(pcm16) / float(sr)
    base_offset = _offset_seconds_for_idx(payload.idx, chunk_len_s)

    # prepare audio float
    audio_f32 = _to_float32(pcm16_full)

    # language strategy:
    #  - default: autodetect (None)
    #  - if mostly German, you can set ASR_FORCE_LANG=de to skip detection and speed up
    force_lang = os.getenv("ASR_FORCE_LANG")  # e.g., "de" or "en"
    lang_arg = force_lang if force_lang else None

    # CPU-friendly decode settings (accuracy/speed balance)
    segments, info = _MODEL.transcribe(
        audio_f32,
        language=lang_arg,                 # None = autodetect; set "de" to force German
        vad_filter=Falser,                   # ignore silence
        beam_size=3,                       # smaller beam for CPU speed
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=True,   # steady continuity
        initial_prompt=_prev_text[-400:] if _prev_text else None,
        word_timestamps=False,             # faster on CPU
        no_speech_threshold=0.5,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    lang = (getattr(info, "language", None) or (force_lang or "en"))

    out = []
    appended_texts = []
    tail_shift = (_TAIL_S / float(sr)) if _tail_pcm.size else 0.0

    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue

        # segment times are relative to pcm16_full (starts with 0.3s of tail)
        seg_start = float(s.start or 0.0)
        seg_end = float(s.end or 0.0)

        # subtract tail, then add chunk base offset
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
        appended_texts.append(text)

    # update streaming context for next call
    _tail_pcm = pcm16[-_TAIL_S:] if pcm16.size > _TAIL_S else pcm16.copy()
    if appended_texts:
        _prev_text = (_prev_text + " " + " ".join(appended_texts)).strip()
        if len(_prev_text) > 2000:
            _prev_text = _prev_text[-2000:]

    return {"segments": out}
