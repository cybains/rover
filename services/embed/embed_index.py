from __future__ import annotations
import faiss


from .chunking import iter_chunks


DATA = Path(__file__).resolve().parents[2] / "data"
DOCS = DATA / "docs.jsonl"
INDEX = DATA / "faiss.index"
META = DATA / "faiss.meta.json"


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"




def load_new_docs() -> List[dict]:
# For MVP we embed all docs each run; swap to a state file of last embedded line later
if not DOCS.exists():
return []
with DOCS.open("r", encoding="utf-8") as f:
return [json.loads(line) for line in f if line.strip()]




def embed_texts(texts: List[str]) -> np.ndarray:
model = SentenceTransformer(MODEL_NAME)
vecs = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
return vecs.astype("float32")




def save_index(index, metas: List[dict]):
faiss.write_index(index, str(INDEX))
META.write_text(json.dumps(metas, ensure_ascii=False), encoding="utf-8")




def load_index(dim: int):
if INDEX.exists():
return faiss.read_index(str(INDEX))
return faiss.IndexFlatIP(dim)




def main():
docs = load_new_docs()
if not docs:
print("no docs to embed")
return


chunks = list(iter_chunks(docs))
if not chunks:
print("no chunks produced")
return


texts = [c["text"] for c in chunks]
vecs = embed_texts(texts)


index = load_index(vecs.shape[1])
metas = []
if META.exists():
metas = json.loads(META.read_text(encoding="utf-8"))


# Append-only for MVP. Later: implement upsert by chunk_id using IDMap2 or IVF+ID.
index.add(vecs)
metas.extend([{k: v for k, v in c.items() if k != "text"} for c in chunks])


save_index(index, metas)
print(f"indexed {len(chunks)} chunks → {INDEX}")




if __name__ == "__main__":
main()