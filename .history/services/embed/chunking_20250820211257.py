from __future__ import annotations
from typing import Dict, Iterable, List
import textwrap




def chunk(core_doc: Dict, max_chars: int = 1000, overlap: int = 120) -> List[Dict]:
text = core_doc["text"]
if not text:
return []
parts: List[str] = []
start = 0
n = len(text)
while start < n:
end = min(start + max_chars, n)
parts.append(text[start:end])
start = end - overlap if end < n else end
if start < 0:
start = 0
chunks = []
for i, p in enumerate(parts):
chunks.append({
"doc_id": core_doc["source_id"],
"chunk_id": f"{core_doc['source_id']}::{i}",
"text": p,
"metadata": {k: core_doc.get(k) for k in ("source","doc_type","country","tags","title")},
})
return chunks




def iter_chunks(normalized_docs: Iterable[Dict]) -> Iterable[Dict]:
for nd in normalized_docs:
core = {k: v for k, v in nd.items() if k not in ("facet", "raw")}
for c in chunk(core):
yield c