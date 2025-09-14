import base64

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from faster_whisper import WhisperModel

app = FastAPI()

_MODEL: WhisperModel | None = None
GERMAN_TOKENS = {"der", "die", "und"}


class TranscribeIn(BaseModel):
    text: str
    idx: int


class ChunkIn(BaseModel):
    audio_b16: str
    sample_rate: int
    idx: int


def _detect_lang(text: str) -> str:
    t = text.lower()
    if any(tok in t.split() for tok in GERMAN_TOKENS) or any(ch in t for ch in "äöüß"):
        return "de"
    return "en"


@app.on_event("startup")
def _load_model():
    global _MODEL
    _MODEL = WhisperModel("small", device="cpu", compute_type="int8")


@app.get("/health")
def health():
    return {"service": "asr", "status": "ok"}


@app.post("/transcribe")
def transcribe(payload: TranscribeIn):
    lang = _detect_lang(payload.text)
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


@app.post("/transcribe_chunk")
def transcribe_chunk(payload: ChunkIn):
    if _MODEL is None:
        return {"segments": []}

    audio_bytes = base64.b64decode(payload.audio_b16)
    samples = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    segs, info = _MODEL.transcribe(samples, language=None)
    lang = info.language if info and info.language else "en"

    out = []
    for s in segs:
        text = (s.text or "").strip()
        if not text:
            continue
        t_start = float(s.start or 0.0)
        t_end = float(s.end or 0.0)
        out.append(
            {
                "idx": payload.idx,
                "tStart": t_start,
                "tEnd": min(t_end, 1.0),
                "lang": lang,
                "speaker": "Speaker 1",
                "textSrc": text,
                "partial": False,
                "confidence": 0.9,
            }
        )

    return {"segments": out}
