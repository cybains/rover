# src/backend/rag_endpoint.py
from __future__ import annotations
import os
from typing import List, Dict, Any
import requests
from flask import Blueprint, request, jsonify

from .rag_retriever import RAGRetriever, Retrieved

rag_bp = Blueprint("rag", __name__)
_retriever: RAGRetriever | None = None

def get_retriever() -> RAGRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RAGRetriever()  # uses repo_root/data by default
    return _retriever

def build_prompt(question: str, chunks: List[Retrieved], max_chars: int = 6000) -> str:
    """
    Builds a compact RAG prompt. Starts with 'You need ...' per your preference.
    """
    header = "You need to answer the user's question using ONLY the provided context. If the answer is not in the context, say you don't know.\n"
    header += "Be concise. Include any key numbers. Cite sources with [#] where # is the context number.\n\n"
    ctx_lines = []
    used = 0
    for i, r in enumerate(chunks, 1):
        piece = f"[{i}] {r.title}\n{r.text}\n"
        if used + len(piece) > max_chars:
            break
        ctx_lines.append(piece)
        used += len(piece)

    ctx = "Context:\n" + "\n".join(ctx_lines) + "\n"
    q = f"Question: {question}\n\n"
    instr = "You need to produce a direct answer followed by a short bullet list of the key facts you used.\n"
    return header + ctx + q + instr

def call_llm(prompt: str) -> str:
    """
    Call YOUR model. Set env var LLM_URL to your generation endpoint.
    Expected JSON: {prompt: str, max_tokens: int, temperature: float} -> returns {text: "..."} or OpenAI-ish.
    Adjust as needed to match your existing Flask API.
    """
    url = os.getenv("LLM_URL")
    if not url:
        # Fallback: make it obvious how to configure
        return "(Set LLM_URL env var to your model endpoint) Prompt preview:\n\n" + prompt[:1200]

    try:
        resp = requests.post(url, json={"prompt": prompt, "max_tokens": 300, "temperature": 0.2}, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # Try common shapes
        if isinstance(data, dict):
            if "text" in data:
                return data["text"]
            if "choices" in data and data["choices"]:
                c0 = data["choices"][0]
                if isinstance(c0, dict):
                    return c0.get("text") or c0.get("message", {}).get("content", "")
        return str(data)
    except Exception as e:
        return f"(LLM call failed: {e})"

@rag_bp.route("/ask", methods=["POST"])
def ask():
    payload: Dict[str, Any] = request.get_json(force=True, silent=True) or {}
    question = (payload.get("question") or "").strip()
    k = int(payload.get("k") or 5)
    if not question:
        return jsonify({"error": "Missing 'question'"}), 400

    retriever = get_retriever()
    hits = retriever.retrieve(question, k=k)
    prompt = build_prompt(question, hits)

    answer = call_llm(prompt)
    citations = []
    for i, r in enumerate(hits, 1):
        citations.append({
            "n": i,
            "source_id": r.source_id,
            "title": r.title,
            "link": r.link
        })

    return jsonify({"answer": answer, "citations": citations})
