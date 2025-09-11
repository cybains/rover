import os, sys, json, re, math
from datetime import datetime
from typing import Any, Dict, List, Optional
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml
from pymongo import MongoClient, UpdateOne

# ---------- PATHS / .ENV ----------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENRICH_CONFIG = ROOT / "config" / "enrichment.yaml"

# Load .env (safe if missing)
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# ---------- ENV ----------
MONGO_URI = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
if not MONGO_URI:
    print("ERROR: Set MONGO_URI (or MONGODB_URI) in your .env")
    sys.exit(1)

DB_NAME = os.environ.get("JOBS_DB_NAME", "refjobs")
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "LLAMACPP")
LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "Phi-3.5-mini-instruct-Q5_K_S.gguf")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
jobs_col = db["jobs"]

# ---------- CONFIG ----------
if not ENRICH_CONFIG.exists():
    print(f"ERROR: enrichment.yaml not found at {ENRICH_CONFIG}")
    sys.exit(1)

with open(ENRICH_CONFIG, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

BATCH_SIZE = int(CFG.get("batch", {}).get("size", 50))
MAX_CONCURRENCY = int(CFG.get("batch", {}).get("max_concurrency", 2))
MAX_JOBS_TOTAL = int(CFG.get("batch", {}).get("max_jobs_total", 0))
PROMPTS = CFG.get("prompts", {}) or {}
MAX_OUT = CFG.get("llm", {}).get("max_output_tokens", {}) or {}
MAX_INPUT_CHARS = int(CFG.get("llm", {}).get("max_input_chars", 8000))

# ---------- HELPERS ----------
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

TAG_RE = re.compile(r"<[^>]+>")
def strip_html(s: Optional[str]) -> str:
    if not s:
        return ""
    return TAG_RE.sub(" ", s)

def clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def compute_quality_score(job_doc: Dict[str, Any], enrich: Dict[str, Any]) -> int:
    score = 0
    # +20 salary present (min or max)
    sal = enrich.get("salary") or {}
    if (sal.get("min") is not None) or (sal.get("max") is not None):
        score += 20
    # +15 apply_url present
    if isinstance(job_doc.get("apply_url"), str) and job_doc["apply_url"].startswith(("http://", "https://")):
        score += 15
    # +20 description length >= 800 chars (strip html first)
    if len(strip_html(job_doc.get("description_html"))) >= 800:
        score += 20
    # +15 >= 5 skills
    skills = enrich.get("skills") or []
    if isinstance(skills, list) and len(skills) >= 5:
        score += 15
    # +10 normalized_title
    if enrich.get("normalized_title"):
        score += 10
    # +10 seniority
    if enrich.get("seniority"):
        score += 10
    # +10 remote_mode
    if enrich.get("remote_mode"):
        score += 10
    # -20 spam_flag
    if enrich.get("spam_flag") is True:
        score -= 20
    return clamp(score, 0, 100)

def read_prompt(path_key: str) -> str:
    path = PROMPTS.get(path_key)
    if not path:
        return ""
    p = ROOT / path
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return ""

def call_llm_json(prompt: str, input_text: str, max_tokens: int) -> Optional[Dict[str, Any]]:
    """
    Calls llama.cpp (OpenAI-compatible /v1/chat/completions) and expects JSON object.
    """
    url = f"{LLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
    sys_prompt = "You are a helpful assistant. Return ONLY valid JSON. No prose, no markdown."
    user_msg = prompt.strip() + "\n\n---\nJOB TEXT:\n" + input_text.strip()
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg}
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens or 256,
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"].strip()
        # Try direct JSON parse; if fails, extract the first {...} block
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if m:
                return json.loads(m.group(0))
            return None
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return None

def build_input_blob(job: Dict[str, Any]) -> str:
    parts = []
    parts.append(f"Title: {job.get('title') or ''}")
    c = job.get("company", {}) or {}
    parts.append(f"Company: {c.get('name') or ''}")
    parts.append(f"Source: {job.get('source') or ''}")
    locs = job.get("locations") or []
    if locs:
        parts.append(f"Locations: {json.dumps(locs, ensure_ascii=False)}")
    # prefer text for LLM
    parts.append("Description:\n" + strip_html(job.get("description_html")))
    text = "\n".join(parts)
    # keep input within limit
    if len(text) > MAX_INPUT_CHARS:
        text = text[:MAX_INPUT_CHARS]
    return text

def enrich_one(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = build_input_blob(job)

    # 1) title + seniority
    p1 = read_prompt("title_seniority") or 'Return JSON: {"normalized_title": "", "seniority": "null"}'
    j1 = call_llm_json(p1, text, MAX_OUT.get("title_seniority", 64)) or {}

    # 2) skills
    p2 = read_prompt("skills") or 'Return JSON: {"skills": []}'
    j2 = call_llm_json(p2, text, MAX_OUT.get("skills", 128)) or {}

    # 3) remote mode
    p3 = read_prompt("remote_mode") or 'Return JSON: {"remote_mode":"onsite"}'
    j3 = call_llm_json(p3, text, MAX_OUT.get("remote_mode", 16)) or {}

    # 4) salary parse
    p4 = read_prompt("salary_extract") or 'Return JSON: {"salary":{"min":null,"max":null,"currency":null,"period":null,"estimated":false}}'
    j4 = call_llm_json(p4, text, MAX_OUT.get("salary_extract", 64)) or {}

    # 5) summary
    p5 = read_prompt("summary") or 'Return JSON: {"summary_tldr": ""}'
    j5 = call_llm_json(p5, text, MAX_OUT.get("summary", 160)) or {}

    # 6) responsibilities
    p6 = read_prompt("responsibilities") or 'Return JSON: {"responsibilities": []}'
    j6 = call_llm_json(p6, text, MAX_OUT.get("responsibilities", 256)) or {}

    # 7) requirements
    p7 = read_prompt("requirements") or 'Return JSON: {"requirements": []}'
    j7 = call_llm_json(p7, text, MAX_OUT.get("requirements", 256)) or {}

    # merge
    enrich: Dict[str, Any] = {
        "normalized_title": j1.get("normalized_title"),
        "seniority": j1.get("seniority"),
        "skills": j2.get("skills"),
        "remote_mode": j3.get("remote_mode"),
        "salary": (j4.get("salary") or {}),
        "summary_tldr": j5.get("summary_tldr"),
        "responsibilities": j6.get("responsibilities"),
        "requirements": j7.get("requirements"),
        # optional defaults
        "spam_flag": False,
    }

    # add quality score
    enrich["quality_score"] = compute_quality_score(job, enrich)
    return enrich

def select_jobs(limit: int) -> List[Dict[str, Any]]:
    """
    Select jobs where enriched_at missing OR last_seen_at > enriched_at.
    """
    q = {
        "$or": [
            {"enriched_at": {"$exists": False}},
            {"$expr": {"$gt": ["$last_seen_at", "$enriched_at"]}}
        ]
    }
    projection = {
        "_id": 1, "title": 1, "company": 1, "source": 1, "locations": 1,
        "description_html": 1, "apply_url": 1, "salary": 1,
        "last_seen_at": 1, "enriched_at": 1
    }
    cur = jobs_col.find(q, projection=projection, limit=limit)
    return list(cur)

def process_batch(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    updated = 0
    ops: List[UpdateOne] = []
    # run with a small thread pool to exercise llama.cpp gently
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        fut_map = {ex.submit(enrich_one, d): d for d in docs}
        for fut in as_completed(fut_map):
            job = fut_map[fut]
            try:
                enrich = fut.result()
            except Exception as e:
                print(f"[ENRICH ERROR] {e}")
                enrich = None
            if not enrich:
                continue
            update_doc = {
                "$set": {
                    "normalized_title": enrich.get("normalized_title"),
                    "seniority": enrich.get("seniority"),
                    "skills": enrich.get("skills"),
                    "remote_mode": enrich.get("remote_mode"),
                    "summary_tldr": enrich.get("summary_tldr"),
                    "responsibilities": enrich.get("responsibilities"),
                    "requirements": enrich.get("requirements"),
                    "quality_score": enrich.get("quality_score"),
                    "spam_flag": enrich.get("spam_flag", False),
                    "enriched_at": now_iso()
                }
            }
            # salary merge: preserve provider salary if present; fill missing/estimated
            sal = enrich.get("salary") or {}
            if any(k in sal for k in ("min", "max", "currency", "period", "estimated")):
                update_doc["$set"]["salary"] = {
                    "min": sal.get("min", job.get("salary", {}).get("min")),
                    "max": sal.get("max", job.get("salary", {}).get("max")),
                    "currency": sal.get("currency", job.get("salary", {}).get("currency")),
                    "period": sal.get("period", job.get("salary", {}).get("period")),
                    "estimated": bool(sal.get("estimated", False))
                }
            ops.append(UpdateOne({"_id": job["_id"]}, update_doc, upsert=False))
            updated += 1
    if ops:
        jobs_col.bulk_write(ops, ordered=False)
    return {"updated": updated}

def main():
    total_to_process = MAX_JOBS_TOTAL if MAX_JOBS_TOTAL > 0 else 10**9
    processed = 0
    while processed < total_to_process:
        left = total_to_process - processed
        batch_limit = min(BATCH_SIZE, left)
        docs = select_jobs(limit=batch_limit)
        if not docs:
            break
        res = process_batch(docs)
        processed += res["updated"]
        print(json.dumps({"batch_done": len(docs), "updated": res["updated"], "processed_total": processed}))
    print(json.dumps({"ok": True, "processed": processed, "db": DB_NAME}, indent=2))

if __name__ == "__main__":
    main()
