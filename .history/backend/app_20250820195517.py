# --- add at top ---
import time, subprocess, os
from typing import Optional

def gpu_mem():
    """Return (total_mb, used_mb) from nvidia-smi or (None, None) if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2
        ).strip().splitlines()[0]
        total, used = [int(x.strip()) for x in out.split(",")]
        return total, used
    except Exception:
        return None, None

# --- add this new Pydantic model + endpoint ---
class BenchReq(BaseModel):
    prompt: Optional[str] = "Summarize Retrieval-Augmented Generation in 3 bullets."
    max_tokens: int = 160
    temperature: float = 0.3
    mode: str = "chat"  # "chat" or "raw"

@app.post("/bench")
def bench(req: BenchReq):
    t0 = time.perf_counter()

    if req.mode == "chat":
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": req.prompt}
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens
        )
        text = out["choices"][0]["message"]["content"]
        usage = out.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(text.split()))
        prompt_tokens = usage.get("prompt_tokens", 0)
    else:
        # raw completion (uses model’s chat template less reliably)
        out = llm(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature
        )
        text = out["choices"][0]["text"]
        # llama.cpp raw completion may not return usage; estimate tokens by length
        completion_tokens = len(text.split())
        prompt_tokens = len(req.prompt.split())

    t1 = time.perf_counter()
    elapsed = t1 - t0
    tok_s = (completion_tokens / elapsed) if elapsed > 0 else None

    total_mb, used_mb = gpu_mem()
    return {
        "elapsed_sec": round(elapsed, 3),
        "tokens_per_sec": round(tok_s, 2) if tok_s else None,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "model_path": os.path.basename(MODEL_PATH),
        "ctx_tokens": CTX_TOKENS,
        "gpu_layers": GPU_LAYERS,
        "gpu_mem_mb": {"total": total_mb, "used": used_mb},
        "sample_output_head": text[:200]
    }





## `backend/app.py`

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
