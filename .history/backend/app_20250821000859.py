# backend/app.py
import os
import time
import subprocess
from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from fastapi import FastAPI
from pydantic import BaseModel
from llama_cpp import Llama

# --- RAG imports ---
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- config from env ----
MODEL_PATH = os.getenv("MODEL_PATH", r"..\models\Phi-3.5-mini-instruct-Q5_K_S.gguf")
CTX_TOKENS = int(os.getenv("CTX_TOKENS", "3072"))
GPU_LAYERS = int(os.getenv("GPU_LAYERS", "36"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))
TOP_P = float(os.getenv("TOP_P", "0.92"))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.08"))

# ---- init model ----
if not os.path.exists(MODEL_PATH):
    raise ValueError(f"Model path does not exist: {MODEL_PATH}")

llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=CTX_TOKENS,
    n_gpu_layers=GPU_LAYERS,
    n_threads=8,
    flash_attn=True,
    offload_kqv=True,
    f16_kv=True,
)

app = FastAPI(title="Rovari Local Model")

# ---- helpers ----
def gpu_mem():
    """Return (total_mb, used_mb) via nvidia-smi or (None, None)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, text=True, timeout=2
        ).strip().splitlines()[0]
        total, used = [int(x.strip()) for x in out.split(",")]
        return total, used
    except Exception:
        return None, None

# ---- schemas ----
class GenReq(BaseModel):
    prompt: str
    max_tokens: int = 256
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repetition_penalty: Optional[float] = None
    mode: str = "raw"  # "raw" or "chat"

class BenchReq(BaseModel):
    prompt: Optional[str] = "Summarize Retrieval-Augmented Generation in 3 bullets."
    max_tokens: int = 160
    temperature: float = 0.3
    mode: str = "chat"  # "chat" or "raw"

# =========================
# RAG: retriever + endpoint
# =========================
DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # <repo_root>/data
IDX_PATH = DATA_DIR / "faiss.index"
META_PATH = DATA_DIR / "meta.jsonl"
DOCS_PATH = DATA_DIR / "docs.jsonl"

_emb_model = None  # lazy-loaded SentenceTransformer
_faiss_index = None
_meta_rows: List[Dict[str, Any]] = []
_texts_by_id: Dict[str, str] = {}


def _load_embeddings_model():
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return _emb_model


def _ensure_index_loaded():
    """Load FAISS index, metadata, and map source_id -> text (once)."""
    global _faiss_index, _meta_rows, _texts_by_id
    if _faiss_index is None:
        if not IDX_PATH.exists() or not META_PATH.exists():
            raise RuntimeError(f"Missing index or meta. Build them first: python tools/build_index.py")
        _faiss_index = faiss.read_index(str(IDX_PATH))

        _meta_rows = []
        with META_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    _meta_rows.append(json.loads(line))

        _texts_by_id = {}
        if DOCS_PATH.exists():
            with DOCS_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        core = rec.get("core", {})
                        sid = core.get("source_id")
                        txt = core.get("text")
                        if sid and isinstance(txt, str):
                            _texts_by_id[sid] = txt
                    except Exception:
                        continue


def _retrieve(query: str, k: int = 5):
    """Return list of dicts: {source_id, title, link, text, score}."""
    _ensure_index_loaded()
    model = _load_embeddings_model()

    q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = _faiss_index.search(q_emb, k)

    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(_meta_rows):
            continue
        m = _meta_rows[idx]
        sid = m.get("source_id")
        title = m.get("title") or sid
        links = m.get("links") or []
        link = links[0] if links else None
        text = _texts_by_id.get(sid, "")
        hits.append({
            "source_id": sid,
            "title": title,
            "link": link,
            "text": text,
            "score": float(score),
        })
    return hits


def _build_prompt(question: str, chunks: List[Dict[str, Any]], max_chars: int = 6000) -> str:
    """Prompt starts with 'You need...' as you requested."""
    header = (
        "You need to answer the user's question using ONLY the provided context. "
        "If the answer is not in the context, say you don't know. "
        "Be concise. Include key numbers. Cite sources as [#].\n\n"
    )
    ctx_lines = []
    used = 0
    for i, r in enumerate(chunks, 1):
        piece = f"[{i}] {r['title']}\n{r['text']}\n"
        if used + len(piece) > max_chars:
            break
        ctx_lines.append(piece)
        used += len(piece)
    ctx = "Context:\n" + "\n".join(ctx_lines) + "\n"
    q = f"Question: {question}\n\n"
    instr = "You need to produce a direct answer followed by a short bullet list of the key facts you used.\n"
    return header + ctx + q + instr


class RAGReq(BaseModel):
    question: str
    k: int = 5
    mode: str = "chat"           # "chat" or "raw"
    max_tokens: int = 300
    temperature: Optional[float] = None


@app.get("/rag/health")
def rag_health():
    ok = IDX_PATH.exists() and META_PATH.exists()
    msg = "ready" if ok else "missing index/meta"
    total_mb, used_mb = gpu_mem()
    return {
        "ok": ok,
        "status": msg,
        "index_path": str(IDX_PATH),
        "meta_path": str(META_PATH),
        "docs_path": str(DOCS_PATH),
        "gpu_mem_mb": {"total": total_mb, "used": used_mb},
    }


@app.post("/rag/ask")
def rag_ask(req: RAGReq):
    hits = _retrieve(req.question, k=req.k)
    prompt = _build_prompt(req.question, hits)

    if req.mode == "chat":
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Be concise and accurate."},
                {"role": "user", "content": prompt},
            ],
            temperature=req.temperature or TEMPERATURE,
            max_tokens=req.max_tokens,
        )
        text = out["choices"][0]["message"]["content"]
    else:
        out = llm(
            prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature or TEMPERATURE,
            top_p=TOP_P,
            repeat_penalty=REPETITION_PENALTY,
        )
        text = out["choices"][0]["text"]

    citations = []
    for i, r in enumerate(hits, 1):
        citations.append({"n": i, "source_id": r["source_id"], "title": r["title"], "link": r["link"]})

    return {"answer": text.strip(), "citations": citations, "used_k": len(hits)}

# =========================
# Existing endpoints
# =========================
@app.get("/health")
def health():
    total_mb, used_mb = gpu_mem()
    return {
        "ok": True,
        "model": os.path.basename(MODEL_PATH),
        "ctx": CTX_TOKENS,
        "gpu_layers": GPU_LAYERS,
        "gpu_mem_mb": {"total": total_mb, "used": used_mb},
    }

@app.post("/generate")
def generate(req: GenReq):
    if req.mode == "chat":
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Be concise and accurate."},
                {"role": "user", "content": req.prompt},
            ],
            temperature=req.temperature or TEMPERATURE,
            max_tokens=req.max_tokens,
        )
        text = out["choices"][0]["message"]["content"]
    else:
        out = llm(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature or TEMPERATURE,
            top_p=req.top_p or TOP_P,
            repeat_penalty=req.repetition_penalty or REPETITION_PENALTY,
        )
        text = out["choices"][0]["text"]

    return {"text": text.strip()}

@app.post("/bench")
def bench(req: BenchReq):
    t0 = time.perf_counter()

    if req.mode == "chat":
        out = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": "Be concise."},
                {"role": "user", "content": req.prompt},
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        text = out["choices"][0]["message"]["content"]
        usage = out.get("usage", {})
        completion_tokens = usage.get("completion_tokens", len(text.split()))
        prompt_tokens = usage.get("prompt_tokens", 0)
    else:
        out = llm(req.prompt, max_tokens=req.max_tokens, temperature=req.temperature)
        text = out["choices"][0]["text"]
        completion_tokens = len(text.split())
        prompt_tokens = len(req.prompt.split())

    elapsed = time.perf_counter() - t0
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
        "sample_output_head": text[:200],
    }
