
## `backend/app.py`
```python
import os
from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

MODEL_PATH = os.getenv("MODEL_PATH", r"..\models\phi-3.5-mini-instruct.Q4_K_M.gguf")
CTX_TOKENS = int(os.getenv("CTX_TOKENS", "3072"))
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "36"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TOP_P", "0.92"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.08"))

# init model
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_TOKENS,
    n_gpu_layers=GPU_LAYERS,   # tune to avoid OOM on 4 GB VRAM
    n_threads=8,               # Ryzen 5500U: try 8–10
    flash_attn=True,
    offload_kqv=True,
    f16_kv=True
)

app = FastAPI(title="Rovari Local Model")

class GenReq(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None

@app.get("/health")
def health():
    return {"ok": True, "model": os.path.basename(MODEL_PATH), "ctx": CTX_TOKENS, "gpu_layers": GPU_LAYERS}

@app.post("/generate")
def generate(req: GenReq):
    params = {
        "max_tokens": req.max_tokens,
        "temperature": req.temperature or TEMPERATURE,
        "top_p": req.top_p or TOP_P,
        "repeat_penalty": req.repetition_penalty or REPETITION_PENALTY
    }
    out = llm(req.prompt, **params)
    return {"text": out["choices"][0]["text"].strip()}
