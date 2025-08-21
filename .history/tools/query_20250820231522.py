# tools/query.py
from __future__ import annotations
import json
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

DATA = Path(__file__).resolve().parents[1] / "data"
IDX  = DATA / "faiss.index"
META = DATA / "meta.jsonl"

def load_meta():
    metas = []
    with META.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                metas.append(json.loads(line))
    return metas

def main():
    import sys
    q = " ".join(sys.argv[1:]) or "What is Austria's GDP in 2023?"
    index = faiss.read_index(str(IDX))
    metas = load_meta()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    q_emb = model.encode([q], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q_emb, 5)
    for rank, (score, idx) in enumerate(zip(D[0], I[0]), 1):
        m = metas[idx]
        print(f"{rank}. {m['title']}  [score={score:.3f}]")
        print(f"   source_id: {m['source_id']}")
        print(f"   link: {m['links'][0] if m['links'] else '—'}")
        print()

if __name__ == "__main__":
    main()
