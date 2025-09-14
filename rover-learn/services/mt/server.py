from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

_MARIAN_NAME = "Helsinki-NLP/opus-mt-de-en"
_NLLB_NAME = "facebook/nllb-200-distilled-600M"
_marian = None
_nllb = None
_LANG_MAP = {"de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn"}


@app.on_event("startup")
def _load_model():
    global _marian, _nllb
    _marian = pipeline("translation", model=_MARIAN_NAME, device=-1)
    _nllb = pipeline("translation", model=_NLLB_NAME, device=-1)


@app.get("/health")
def health():
    return {"service": "mt", "status": "ok"}


class TranslateIn(BaseModel):
    text: str
    src_lang: str | None = None
    tgt_lang: str

def _translate(text: str, src_lang: str, max_length: int) -> str:
    if src_lang == "de":
        out = _marian(text, max_length=max_length, num_beams=1)
        return out[0]["translation_text"]
    src_code = _LANG_MAP.get(src_lang, src_lang)
    out = _nllb(text, src_lang=src_code, tgt_lang="eng_Latn", max_length=max_length, num_beams=1)
    return out[0]["translation_text"]


@app.post("/translate_partial")
def translate_partial(payload: TranslateIn):
    return {"translation": _translate(payload.text, payload.src_lang or "de", 128), "mode": "partial"}


@app.post("/translate_final")
def translate_final(payload: TranslateIn):
    return {"translation": _translate(payload.text, payload.src_lang or "de", 256), "mode": "final"}
