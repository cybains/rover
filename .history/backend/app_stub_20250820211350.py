from __future__ import annotations
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer


DATA = Path(__file__).resolve().parents[1] / "data"
INDEX = faiss.read_index(str(DATA / "faiss.index"))
METAS = json.loads((DATA / "faiss.meta.json").read_text(encoding="utf-8"))
EMB = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


app = FastAPI(title="Rover Search", version="0.1.0")


class SearchReq(BaseModel):
q: str
k: int = 5


@app.get("/healthz")
def health():
return {"ok": True}


@app.post("/search/query")
def search(req: SearchReq):
qv = EMB.encode([req.q], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
D, I = INDEX.search(qv, req.k)
hits = []
for j, score in zip(I[0], D[0]):
m = METAS[j]
hits.append({**m, "score": float(score)})
return {"hits": hits}