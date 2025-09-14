from fastapi import FastAPI
from pydantic import BaseModel
from transformers import MarianMTModel, MarianTokenizer

app = FastAPI()

_MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"
_tokenizer = None
_model = None


@app.on_event("startup")
def _load_model():
    global _tokenizer, _model
    _tokenizer = MarianTokenizer.from_pretrained(_MODEL_NAME)
    _model = MarianMTModel.from_pretrained(_MODEL_NAME)


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
    inputs = _tokenizer(payload.text, return_tensors="pt")
    gen = _model.generate(**inputs)
    out = _tokenizer.decode(gen[0], skip_special_tokens=True)
    return {"translation": out}
