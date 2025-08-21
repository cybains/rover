# src/backend/rag_retriever.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import faiss
import numpy as np

# Lazy import the embedding model to keep API startup snappy
_sentence_model = None

@dataclass
class Retrieved:
    source_id: str
    title: str
    link: Optional[str]
    text: str
    score: float

class RAGRetriever:
    def __init__(self, data_dir: Optional[Path] = None, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).resolve().parents[2] / "data"
        self.idx_path = self.data_dir / "faiss.index"
        self.meta_path = self.data_dir / "meta.jsonl"
        self.docs_path = self.data_dir / "docs.jsonl"
        self.model_name = model_name

        if not self.idx_path.exists() or not self.meta_path.exists():
            raise RuntimeError(f"Missing index or meta. Build with: python tools/build_index.py\nExpected: {self.idx_path}, {self.meta_path}")

        # Load FAISS + metas
        self.index = faiss.read_index(str(self.idx_path))
        self.metas: List[Dict[str, Any]] = []
        with self.meta_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.metas.append(json.loads(line))

        # Map source_id -> core.text (so we can give the LLM the actual context)
        self.texts_by_id: Dict[str, str] = {}
        if self.docs_path.exists():
            with self.docs_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        rec = json.loads(line)
                        core = rec.get("core", {})
                        sid = core.get("source_id")
                        txt = core.get("text")
                        if sid and isinstance(txt, str):
                            self.texts_by_id[sid] = txt
                    except Exception:
                        continue

    def _get_model(self):
        global _sentence_model
        if _sentence_model is None:
            from sentence_transformers import SentenceTransformer
            _sentence_model = SentenceTransformer(self.model_name, device="cpu")
        return _sentence_model

    def retrieve(self, query: str, k: int = 5) -> List[Retrieved]:
        model = self._get_model()
        q_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
        D, I = self.index.search(q_emb, k)
        out: List[Retrieved] = []
        for score, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metas):
                continue
            m = self.metas[idx]
            sid = m.get("source_id")
            title = m.get("title") or sid
            links = m.get("links") or []
            link = links[0] if links else None
            text = self.texts_by_id.get(sid, "")
            out.append(Retrieved(source_id=sid, title=title, link=link, text=text, score=float(score)))
        return out
