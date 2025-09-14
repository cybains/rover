from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

GERMAN_TOKENS = {"der", "die", "und"}


class TranscribeIn(BaseModel):
    text: str
    idx: int


def _detect_lang(text: str) -> str:
    t = text.lower()
    if any(tok in t.split() for tok in GERMAN_TOKENS) or any(ch in t for ch in "äöüß"):
        return "de"
    return "en"


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
