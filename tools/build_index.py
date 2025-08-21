# tools/build_index.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import List, Dict
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
DOCS = DATA / "docs.jsonl"
IDX  = DATA / "faiss.index"
META = DATA / "meta.jsonl"

def iter_docs():
    with DOCS.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def main():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts: List[str] = []
    metas: List[Dict] = []

    for rec in iter_docs():
        core = rec["core"]
        text = core.get("text") or ""
        if not text.strip():
            continue
        texts.append(text)
        metas.append({
            "source_id": core.get("source_id"),
            "title": core.get("title"),
            "links": core.get("links"),
            "tags": core.get("tags"),
            "country": core.get("country"),
        })

    if not texts:
        print("No texts found in docs.jsonl")
        return

    embs = model.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embs.astype(np.float32))

    faiss.write_index(index, str(IDX))
    with META.open("w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"Indexed {len(texts)} docs → {IDX}, metadata → {META}")

if __name__ == "__main__":
    main()
