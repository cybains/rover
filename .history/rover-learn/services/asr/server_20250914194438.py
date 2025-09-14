# rover-learn/services/asr/server.py
from __future__ import annotations

import base64
import os
from typing import Optional, Tuple

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel

app = FastAPI()

# ---------- Model & Streaming Context ----------

_MODEL: Optional[WhisperModel] = None

# simple single-stream context (good for one user at a time)
_SR = 16000                     # expected sample rate
_TAIL_S = int(0.3 * _SR)        # 300 ms overlap
_tail_pcm = np.zeros(0, dtype=np.int16)
_prev_text: str = ""            # rolling text prompt memory


def _init_model() -> WhisperModel:
    """
    Prefer GPU with low-VRAM settings; fallback to CPU.
    You can override via env:
      ASR_MODEL      (default: distil-large-v3 or small)
      ASR_DEVICE     (default: auto -> cuda if available else cpu)
      ASR_COMPUTE    (default: int8_float16 for cuda, int8 for cpu)
    """
    model_id = os.getenv("ASR_MODEL") or "distil-large-v3"
    # If distil-large-v3 is unavailable in your env, "small" is a safe choice:
    # model_id = os.getenv("ASR_MODEL") or "small"

    # decide device
    dflt_device = "cuda" if os.getenv("CUDA_VISIBLE_DEVICES", "") != "" else "cuda"
    # Let faster-whisper handle non-CUDA gracefully; we'll still try cuda first
    device = os.getenv("ASR_DEVICE", dflt_device)

    # compute type tuned for VRAM (GTX 1650 works well with int8_float16)
    compute_type = os.getenv("ASR_COMPUTE")
    if compute_type is None:
        compute_type = "int8_float16" if device == "cuda" else "int8"

    return WhisperModel(
        model_id,
        device=device,
        compute_type=compute_type,
        cpu_threads=8,   # helps if it falls back to CPU
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
    # (optional) if you add this later, we can keep per-session context
    # sessionId: Optional[str] = None


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
    # Place this chunk on a global timeline by index
    return max(0.0, float(idx) * float(chunk_len_s))


# ---------- FastAPI Lifecycle ----------

@app.on_event("startup")
def _load_model():
    global _MODEL
    _MODEL = _init_model()


@app.get("/health")
def health():
    return {"service": "asr", "status": "ok"}


# ---------- Simple text-based path (kept for compatibility) ----------

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
    proper global timestamps based on chunk index and true chunk duration.
    """
    global _MODEL, _tail_pcm, _prev_text

    if _MODEL is None:
        return {"segments": []}

    # basic validation
    sr = int(payload.sample_rate)
    if sr <= 0:
        return {"segments": []}

    # decode to PCM16
    try:
        audio_bytes = base64.b64decode(payload.audio_b16)
    except Exception:
        return {"segments": []}

    pcm16 = np.frombuffer(audio_bytes, dtype=np.int16)

    # Ensure mono (agent already downmixes; if not, downmix here)
    if pcm16.ndim != 1:
        pcm16 = pcm16.reshape(-1)

    # append overlap tail
    if _tail_pcm.size:
        pcm16_full = np.concatenate([_tail_pcm, pcm16])
    else:
        pcm16_full = pcm16

    # compute chunk length (based on *current* chunk, not including tail)
    chunk_len_s = len(pcm16) / float(sr)
    base_offset = _offset_seconds_for_idx(payload.idx, chunk_len_s)

    # prepare audio float
    audio_f32 = _to_float32(pcm16_full)

    # decode with accuracy/speed tuned for 4GB VRAM GPUs (or CPU fallback)
    # NOTE: We *let Whisper detect language* (None) so it works for any language
    # If you mostly do German, you can pass language="de" to speed up a bit.
    segments, info = _MODEL.transcribe(
        audio_f32,
        language=None,  # autodetect; set "de" to force German if desired
        vad_filter=True,
        beam_size=5,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=True,
        initial_prompt=_prev_text[-400:] if _prev_text else None,
        word_timestamps=False,
        no_speech_threshold=0.5,
        log_prob_threshold=-1.0,
        compression_ratio_threshold=2.4,
    )

    lang = (info.language if info and getattr(info, "language", None) else "en")

    out = []
    appended_texts = []
    for s in segments:
        text = (s.text or "").strip()
        if not text:
            continue

        # segment times are relative to pcm16_full (which starts with 0.3s of tail)
        # shift to global timeline by removing tail offset and adding base_offset
        seg_start = float(s.start or 0.0)
        seg_end = float(s.end or 0.0)

        # subtract tail portion (so times align to current window), then add chunk base
        tail_shift = _TAIL_S / float(sr) if _tail_pcm.size else 0.0
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
    # keep last 300ms tail from *this* raw chunk (not full)
    _tail_pcm = pcm16[-_TAIL_S:] if pcm16.size > _TAIL_S else pcm16.copy()
    # extend rolling prompt with new clean text
    if appended_texts:
        _prev_text = (_prev_text + " " + " ".join(appended_texts)).strip()
        # limit memory to avoid unbounded growth
        if len(_prev_text) > 2000:
            _prev_text = _prev_text[-2000:]

    return {"segments": out}
