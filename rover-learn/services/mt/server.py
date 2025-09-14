from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

_MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
_nlp = None


@app.on_event("startup")
def _load_model():
    global _nlp
    _nlp = pipeline("translation", model=_MODEL_NAME, device=-1)


@app.get("/health")
def health():
    return {"service": "mt", "status": "ok"}


class TranslateIn(BaseModel):
    text: str
    src_lang: str | None = None
    tgt_lang: str


@app.post("/translate")
def translate(payload: TranslateIn):
    if payload.tgt_lang != "en" or payload.src_lang != "de":
        return {"translation": payload.text}
    out = _nlp(payload.text, max_length=256, num_beams=1)
    return {"translation": out[0]["translation_text"]}
