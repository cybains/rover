from __future__ import annotations

from typing import List

from bson import ObjectId
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


async def recompute_qalinks(db, session_id: str) -> None:
    try:
        oid = ObjectId(session_id)
        query = {"sessionId": {"$in": [session_id, oid]}}
    except Exception:
        query = {"sessionId": session_id}

    cursor = db.segments.find(query).sort("idxStart", 1)
    segments: List[dict] = []
    async for seg in cursor:
        segments.append(seg)

    for seg in segments:
        if not seg.get("isQuestion"):
            continue
        q_text = (seg.get("textSrc") or "").strip()
        if not q_text:
            continue
        q_idx = seg.get("idxStart")
        q_time = seg.get("tStart", 0.0)

        cands = [s for s in segments if s.get("idxStart") != q_idx and abs((s.get("tStart") or 0) - q_time) <= 300]
        if not cands:
            cands = [s for s in segments if s.get("idxStart") != q_idx and abs(s.get("idxStart", 0) - q_idx) <= 50]
        if not cands:
            await db.segments.update_one({"_id": seg["_id"]}, {"$unset": {"qa": ""}})
            continue

        texts = [q_text] + [(c.get("textSrc") or "").strip() for c in cands]
        try:
            vec = TfidfVectorizer().fit_transform(texts)
            sims = cosine_similarity(vec[0:1], vec[1:]).flatten()
        except ValueError:
            sims = []
        order = list(range(len(cands)))
        if sims.size:
            order = list(sims.argsort()[::-1])
        cand_idxs = [cands[i].get("idxStart") for i in order[:3]]
        best_idx = cand_idxs[0] if cand_idxs else None
        await db.segments.update_one(
            {"_id": seg["_id"]},
            {"$set": {"qa": {"bestAnswerIdx": best_idx, "candidateAnswerIdxs": cand_idxs}}},
        )
