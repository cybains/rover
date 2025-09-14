import os, sys, json, re, math, html
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml
from pymongo import MongoClient, UpdateOne

# -------- Optional deps (graceful fallback) --------
try:
    from bs4 import BeautifulSoup  # better HTML cleaner
except Exception:
    BeautifulSoup = None

try:
    # fast + reasonable language id; if missing, we fallback to simple heuristics
    from langdetect import detect as lang_detect
except Exception:
    lang_detect = None

# ---------- PATHS / .ENV ----------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENRICH_CONFIG = ROOT / "config" / "enrichment_v2.yaml"

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

# LLM (llama.cpp OpenAI-compat)
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "LLAMACPP")
LLAMA_BASE_URL = os.environ.get("LLAMA_BASE_URL", "http://127.0.0.1:8080")
LLAMA_MODEL = os.environ.get("LLAMA_MODEL", "Phi-3.5-mini-instruct-Q5_K_S.gguf")

# FX (optional): fixed table or service; keep simple placeholder
FX_USD_RATES = {"EUR": 1.08, "GBP": 1.27, "USD": 1.0, "CHF": 1.12}

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
jobs_col = db["jobs"]

# ---------- CONFIG ----------
if not ENRICH_CONFIG.exists():
    print(f"ERROR: enrichment_v2.yaml not found at {ENRICH_CONFIG}")
    sys.exit(1)

with open(ENRICH_CONFIG, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f) or {}

BATCH_SIZE = int(CFG.get("batch", {}).get("size", 50))
MAX_CONCURRENCY = int(CFG.get("batch", {}).get("max_concurrency", 2))
MAX_JOBS_TOTAL = int(CFG.get("batch", {}).get("max_jobs_total", 0))

PROMPTS = CFG.get("prompts", {}) or {}
MAX_OUT = CFG.get("llm", {}).get("max_output_tokens", {}) or {}
MAX_INPUT_CHARS = int(CFG.get("llm", {}).get("max_input_chars", 8000))
TEMPERATURE = float(CFG.get("llm", {}).get("temperature", 0.2))

LANG_CFG = CFG.get("language", {}) or {}
GUARDS = CFG.get("guardrails", {}) or {}
SCORING = CFG.get("scoring", {}) or {}

ONTOLOGY = CFG.get("ontology", {}) or {}
SKILL_ALLOWLIST = set([s.lower() for s in ONTOLOGY.get("skills_allowlist", [])])
TITLE_MAP = ONTOLOGY.get("title_map", {}) or {}
REMOTE_PATTERNS = [re.compile(p, re.I) for p in ONTOLOGY.get("remote_patterns", [])]
HYBRID_PATTERNS = [re.compile(p, re.I) for p in ONTOLOGY.get("hybrid_patterns", [])]
ONSITE_PATTERNS = [re.compile(p, re.I) for p in ONTOLOGY.get("onsite_patterns", [])]
CEFR_PATTERNS = {k: re.compile(v, re.I) for k, v in (ONTOLOGY.get("cefr_patterns", {}) or {}).items()}

# ---------- UTILS ----------
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))

def clean_html(html_str: Optional[str]) -> str:
    if not html_str:
        return ""
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html_str, "lxml")
            for s in soup(["script", "style"]):
                s.extract()
            text = soup.get_text(" ")
        except Exception:
            text = re.sub(r"<[^>]+>", " ", html_str)
    else:
        text = re.sub(r"<[^>]+>", " ", html_str)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text, flags=re.MULTILINE).strip()
    return text

def checksum_text(s: str) -> str:
    # Small stable checksum to detect content changes; avoids extra deps
    import hashlib
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def detect_language(text: str) -> Tuple[str, float]:
    if not text or len(text) < 40:
        return ("en", 0.5)
    if lang_detect:
        try:
            code = lang_detect(text)
            return (code[:2], 0.95)
        except Exception:
            pass
    # tiny heuristic fallback
    if re.search(r"[äöüß]", text):
        return ("de", 0.7)
    if re.search(r"[àâçéèêëîïôùûüÿœ]", text):
        return ("fr", 0.6)
    if re.search(r"[áâãàçéêíóôõú]", text):
        return ("pt", 0.6)
    return ("en", 0.5)

def mt_translate_to_en(text: str, src_lang: str) -> Tuple[str, float, str]:
    """
    Placeholder MT hook: returns original if English or MT disabled.
    Wire this to your MT service later; keep same shape (text_en, qe_score, provider).
    """
    if not text:
        return ("", 0.0, "")
    prefer = set(LANG_CFG.get("prefer", []) or [])
    if src_lang == "en" or not LANG_CFG.get("detect", True):
        return (text, 1.0, "")
    # If you add real MT, call here and return (translated, qe_score, "provider")
    return (text, 0.7, "noop")  # fallback: use original; low QE so we gate risky inferences

def usd_band(min_amt: Optional[float], max_amt: Optional[float], currency: Optional[str], period: Optional[str]) -> Dict[str, Any]:
    if not currency or currency.upper() not in FX_USD_RATES:
        return {}
    rate = FX_USD_RATES[currency.upper()]
    conv = lambda x: round(float(x) * rate, 2) if x is not None else None
    return {"min": conv(min_amt), "max": conv(max_amt), "currency": "USD", "period": period}

def norm_list_str(xs: Optional[List[str]], lower=True) -> List[str]:
    if not xs:
        return []
    seen, out = set(), []
    for x in xs:
        if not isinstance(x, str):
            continue
        v = x.strip()
        if not v:
            continue
        v2 = v.lower() if lower else v
        if v2 not in seen:
            seen.add(v2)
            out.append(v2)
    return out

# ---------- QUALITY / SCORING ----------
def compute_quality_score(job_doc: Dict[str, Any], enrich: Dict[str, Any]) -> int:
    """
    v2: adds coherence penalty and rich-lists bonus (configurable).
    """
    base = 0
    sal = enrich.get("salary") or {}
    if (sal.get("min") is not None) or (sal.get("max") is not None):
        base += 20
    if isinstance(job_doc.get("apply_url"), str) and job_doc["apply_url"].startswith(("http://", "https://")):
        base += 15
    if len(clean_html(job_doc.get("description_html"))) >= 800:
        base += 20
    skills = enrich.get("skills") or []
    if isinstance(skills, list) and len(skills) >= 5:
        base += 15
    if enrich.get("normalized_title"):
        base += 10
    if enrich.get("seniority"):
        base += 10
    if enrich.get("remote_mode"):
        base += 10
    if enrich.get("spam_flag") is True:
        base -= 20
    # coherence penalty
    if enrich.get("remote_mode") in {"remote", "hybrid"} and job_doc.get("locations"):
        # If locations list is obviously onsite-only with nulls, penalize (can be tuned)
        locs = job_doc.get("locations") or []
        if any((isinstance(l, dict) and l.get("type") == "onsite" and not any(l.get(k) for k in ("city", "region", "country")) ) for l in locs):
            base -= int(SCORING.get("coherence_penalty", 10))
    # lists bonus
    if len(enrich.get("requirements") or []) >= 4 and len(enrich.get("responsibilities") or []) >= 4:
        base += int(SCORING.get("rich_lists_bonus", 5))
    return int(clamp(base, 0, 100))

# ---------- DETERMINISTIC EXTRACTORS ----------
def derive_remote(text: str, raw: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Returns: (remote_bool, remote_mode_enum, remote_scope_dict)
    """
    t = text.lower()
    mode = "onsite"
    scope: Dict[str, Any] = {}
    remote = False

    if any(p.search(t) for p in REMOTE_PATTERNS):
        remote, mode = True, "remote"
    if any(p.search(t) for p in HYBRID_PATTERNS):
        mode = "hybrid"
        remote = True
    if any(p.search(t) for p in ONSITE_PATTERNS):
        mode = "onsite"
        remote = False

    # provider hints (e.g., candidate_required_location)
    loc = (raw or {}).get("candidate_required_location")
    if loc:
        # simple country-only capture (extend later)
        scope["country"] = loc.strip()

    return (remote, mode, scope)

def title_normalize(raw_title: str) -> Tuple[str, str]:
    """
    Map raw titles to a normalized title & seniority using small rules + TITLE_MAP.
    """
    if not raw_title:
        return ("", "")
    rt = raw_title.lower().strip()
    # seniority
    seniority = ""
    if re.search(r"\b(intern|working student)\b", rt): seniority = "intern"
    elif re.search(r"\b(junior|jr\.)\b", rt): seniority = "junior"
    elif re.search(r"\b(senior|sr\.)\b", rt): seniority = "senior"
    elif re.search(r"\b(lead|principal|staff)\b", rt): seniority = "lead"
    elif re.search(r"\b(manager|head)\b", rt): seniority = "manager"

    # title map exact/contains
    for k, v in TITLE_MAP.items():
        if re.search(k, rt, flags=re.I):
            return (v, seniority or "")
    # fallback: strip punctuation and collapse
    rt = re.sub(r"[^a-z0-9 ]+", " ", rt)
    rt = re.sub(r"\s+", " ", rt).strip()
    return (rt.replace(" ", "_"), seniority or "")

def seed_skills_from_text_and_tags(text: str, provider_tags: List[str]) -> List[str]:
    seeds = set()
    for t in (provider_tags or []):
        tt = t.strip().lower()
        if tt and (not SKILL_ALLOWLIST or tt in SKILL_ALLOWLIST):
            seeds.add(tt)
    # exact word boundary matches from allowlist (avoid hallucinations)
    if SKILL_ALLOWLIST:
        for sk in SKILL_ALLOWLIST:
            if re.search(rf"\b{re.escape(sk)}\b", text, flags=re.I):
                seeds.add(sk)
    return sorted(seeds)

def parse_cefr_languages(text: str) -> List[Dict[str, Any]]:
    out = []
    for level, rx in CEFR_PATTERNS.items():
        if rx.search(text):
            # crude language guess from surrounding text
            # you can expand later; for now, tag the level without language id
            out.append({"lang": "", "level": level, "source": "desc", "confidence": 0.7})
    return out

# ---------- LLM ----------
def call_llm_json(single_prompt: str, input_text: str, max_tokens: int) -> Optional[Dict[str, Any]]:
    url = f"{LLAMA_BASE_URL.rstrip('/')}/v1/chat/completions"
    sys_prompt = (
        "You are an information extraction service. "
        "Return ONLY minified JSON that strictly follows the schema. No prose, no markdown."
    )
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": single_prompt + "\n\n---\nJOB TEXT:\n" + input_text.strip()}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens or 512,
    }
    try:
        r = SESSION.post(url, json=payload, timeout=CFG.get("timeouts", {}).get("llm", 120))
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"].strip()
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

def build_single_prompt(skills_seed: List[str]) -> str:
    """
    Single batched prompt to get: responsibilities, requirements, summary_tldr,
    remote_mode (as cross-check), salary (optional), and skill augmentation (bounded).
    We pass the allowlist and seeds to reduce hallucinations.
    """
    p = (Path(ROOT / (PROMPTS.get("batched") or "")).read_text(encoding="utf-8")
         if PROMPTS.get("batched") else "")
    if not p:
        # safe fallback prompt
        p = (
            "Extract fields as JSON with keys: "
            '{"responsibilities":[], "requirements":[], "summary_tldr":"", '
            '"remote_mode":"", "salary":{"min":null,"max":null,"currency":null,"period":null,"estimated":false}, '
            '"skills_extra":[]}. '
            "Use only skills from ALLOWLIST if present in text; do not invent.\n"
            f"ALLOWLIST={sorted(list(SKILL_ALLOWLIST))}\n"
            f"SEED_SKILLS={sorted(list(skills_seed))}\n"
            "Rules: arrays max 8 items; summary <= 240 chars; remote_mode in [remote,hybrid,onsite]; "
            "salary numbers only if explicitly present; skills_extra only from allowlist with evidence in text."
        )
    return p

# ---------- VALIDATION ----------
SENIORITY_ENUM = {"intern","junior","mid","senior","lead","manager","director","vp","cxo"}
REMOTE_ENUM = {"remote","hybrid","onsite"}

def validate_enriched(doc: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(doc)
    # enums
    if out.get("seniority") and out["seniority"] not in SENIORITY_ENUM:
        out["seniority"] = ""
    if out.get("remote_mode") and out["remote_mode"] not in REMOTE_ENUM:
        out["remote_mode"] = ""
    # lists
    for k in ("skills","responsibilities","requirements"):
        out[k] = norm_list_str(out.get(k))
    # salary
    sal = out.get("salary") or {}
    for key in ("min","max"):
        v = sal.get(key)
        if v is not None:
            try:
                sal[key] = float(v)
                if sal[key] < 0: sal[key] = None
            except Exception:
                sal[key] = None
    cur = sal.get("currency")
    if cur and not re.fullmatch(r"[A-Z]{3}", str(cur).upper()):
        sal["currency"] = None
    out["salary"] = sal
    # cap list sizes to keep payload tidy
    out["requirements"] = out["requirements"][:8]
    out["responsibilities"] = out["responsibilities"][:8]
    out["skills"] = out["skills"][:20]
    return out

# ---------- SESSION (keep-alive) ----------
SESSION = requests.Session()

# ---------- ENRICH ----------
def build_input(job: Dict[str, Any]) -> Dict[str, Any]:
    raw = job.get("raw") or {}
    desc_html = job.get("description_html") or ""
    text_original = clean_html(desc_html)
    text_ck = checksum_text(text_original) if text_original else ""

    # language
    lang_code, lang_conf = ("en", 1.0)
    text_en = text_original
    if LANG_CFG.get("detect", True):
        lang_code, lang_conf = detect_language(text_original)
        text_en, qe, mt_provider = mt_translate_to_en(text_original, lang_code)
    else:
        qe, mt_provider = (1.0, "")

    # base fields
    base = {
        "title": job.get("title") or "",
        "company_name": (job.get("company") or {}).get("name") or "",
        "source": job.get("source") or "",
        "raw": raw,
        "text_original": text_original,
        "text_en": text_en[:MAX_INPUT_CHARS] if text_en else "",
        "lang_detected": {"code": lang_code, "confidence": lang_conf},
        "translation": {"status": ("mt" if lang_code != "en" else "none"),
                        "qe_score": qe, "provider": mt_provider},
        "text_checksum": text_ck,
        "provider_tags": raw.get("tags") or job.get("tags") or [],
    }
    return base

def enrich_one(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    meta = build_input(job)
    text_for_extract = meta["text_en"] or meta["text_original"]
    if not text_for_extract:
        return None

    # deterministic: title + seniority
    nt, sen = title_normalize(job.get("title") or "")
    # deterministic: remote & scope
    remote_bool, remote_mode_rule, remote_scope = derive_remote(text_for_extract, job.get("raw") or {})
    # seed skills grounded in provider tags + allowlist word matches
    seed_skills = seed_skills_from_text_and_tags(text_for_extract, meta["provider_tags"])
    # languages / CEFR basics from original (best recall)
    languages_required = parse_cefr_languages(meta["text_original"])

    # single LLM call for augmentation (bounded, optional if QE low)
    llm_out = {}
    if meta["translation"]["qe_score"] >= float(CFG.get("language",{}).get("qe_min_for_llm", 0.55)):
        prompt = build_single_prompt(seed_skills)
        llm_out = call_llm_json(prompt, text_for_extract, MAX_OUT.get("batched", 512)) or {}
    else:
        llm_out = {}

    # merge skills: seeds + vetted extras from LLM (only allowlist and must appear in text)
    llm_skills_extra = [s for s in (llm_out.get("skills_extra") or []) if isinstance(s, str)]
    llm_skills_extra = [s.lower().strip() for s in llm_skills_extra
                        if s and (not SKILL_ALLOWLIST or s.lower().strip() in SKILL_ALLOWLIST)
                        and re.search(rf"\b{re.escape(s)}\b", text_for_extract, re.I)]
    skills_final = norm_list_str(list(set(seed_skills).union(llm_skills_extra)))

    # salary: prefer LLM only if explicit; otherwise provider/job.salary
    salary_llm = llm_out.get("salary") or {}
    salary_final = {
        "min": salary_llm.get("min"),
        "max": salary_llm.get("max"),
        "currency": (salary_llm.get("currency") or (job.get("salary") or {}).get("currency")),
        "period": (salary_llm.get("period") or (job.get("salary") or {}).get("period")),
        "estimated": bool(salary_llm.get("estimated", False))
    }
    # USD band (optional)
    if salary_final.get("currency"):
        salary_usd = usd_band(salary_final.get("min"), salary_final.get("max"),
                              salary_final.get("currency"), salary_final.get("period"))
    else:
        salary_usd = {}

    # remote_mode final: prefer deterministic rules; use LLM as tie-breaker only
    remote_mode_llm = (llm_out.get("remote_mode") or "").lower().strip()
    remote_mode = remote_mode_rule or (remote_mode_llm if remote_mode_llm in REMOTE_ENUM else "")
    remote_bool = (remote_mode in {"remote","hybrid"}) or remote_bool

    # Assemble enriched doc (with provenance)
    enriched: Dict[str, Any] = {
        "schema_version": "v2",
        "normalized_title": nt,
        "seniority": sen,
        "skills": skills_final,
        "remote": remote_bool,
        "remote_mode": remote_mode,
        "remote_scope": remote_scope or {},
        "summary_tldr": (llm_out.get("summary_tldr") or "")[:240],
        "responsibilities": norm_list_str(llm_out.get("responsibilities")),
        "requirements": norm_list_str(llm_out.get("requirements")),
        "salary": salary_final,
        "salary_usd": salary_usd or {},
        "languages": languages_required,  # can be augmented later
        "provenance": {
            "normalized_title": {"source": "rule+map", "confidence": 0.9},
            "seniority": {"source": "rule", "confidence": 0.8 if sen else 0.5},
            "skills": {"source": "tags+text+llm_filtered", "confidence": 0.85},
            "remote_mode": {"source": "rule>llm", "confidence": 0.85},
            "summary_tldr": {"source": "llm", "confidence": 0.7 if llm_out else 0.0},
            "requirements": {"source": "llm", "confidence": 0.7 if llm_out else 0.0},
            "responsibilities": {"source": "llm", "confidence": 0.7 if llm_out else 0.0},
            "salary": {"source": "llm|provider", "confidence": 0.6 if salary_final.get("min") or salary_final.get("max") else 0.0},
            "languages": {"source": "desc", "confidence": 0.7 if languages_required else 0.0},
            "text": {"source": "cleaner", "confidence": 1.0, "lang": meta["lang_detected"], "translation": meta["translation"]},
        },
        "spam_flag": False,
    }

    enriched = validate_enriched(enriched)
    enriched["quality_score"] = compute_quality_score(job, enriched)
    return enriched

# ---------- SELECT & WRITE ----------
def selector_query(limit: int):
    rule = (CFG.get("selector", {}) or {}).get("rule", "enriched_at_missing_or_stale")
    if rule == "enriched_at_missing_or_stale":
        q = {"$or":[{"enriched_at": {"$exists": False}}, {"$expr": {"$gt": ["$last_seen_at", "$enriched_at"]}}]}
    else:
        q = {}
    projection = {
        "_id": 1, "title": 1, "company": 1, "source": 1, "locations": 1,
        "description_html": 1, "apply_url": 1, "salary": 1, "raw": 1,
        "last_seen_at": 1, "enriched_at": 1
    }
    cur = jobs_col.find(q, projection=projection, limit=limit)
    return list(cur)

def process_batch(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    updated = 0
    ops: List[UpdateOne] = []
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

            # Build update doc
            set_fields = {
                "schema_version": "v2",
                "normalized_title": enrich.get("normalized_title"),
                "seniority": enrich.get("seniority"),
                "skills": enrich.get("skills"),
                "remote": enrich.get("remote"),
                "remote_mode": enrich.get("remote_mode"),
                "remote_scope": enrich.get("remote_scope"),
                "summary_tldr": enrich.get("summary_tldr"),
                "responsibilities": enrich.get("responsibilities"),
                "requirements": enrich.get("requirements"),
                "languages": enrich.get("languages"),
                "quality_score": enrich.get("quality_score"),
                "provenance": enrich.get("provenance"),
                "spam_flag": enrich.get("spam_flag", False),
                "enriched_at": now_iso()
            }

            # salary merge: preserve provider salary if present; fill missing/estimated
            job_sal = job.get("salary") or {}
            sal = enrich.get("salary") or {}
            if any(k in sal for k in ("min", "max", "currency", "period", "estimated")) or job_sal:
                set_fields["salary"] = {
                    "min": sal.get("min", job_sal.get("min")),
                    "max": sal.get("max", job_sal.get("max")),
                    "currency": sal.get("currency", job_sal.get("currency")),
                    "period": sal.get("period", job_sal.get("period")),
                    "estimated": bool(sal.get("estimated", False))
                }
            usd = enrich.get("salary_usd") or {}
            if usd:
                set_fields["salary_usd"] = usd

            # Clean up bogus onsite location if remote
            if set_fields.get("remote") and set_fields.get("remote_mode") in {"remote","hybrid"}:
                if job.get("locations"):
                    # remove null-only onsite stubs
                    pass  # keep original locations unless obviously junk; consider clearing in a later migration

            ops.append(UpdateOne({"_id": job["_id"]}, {"$set": set_fields}, upsert=False))
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
        docs = selector_query(limit=batch_limit)
        if not docs:
            break
        res = process_batch(docs)
        processed += res["updated"]
        print(json.dumps({"batch_done": len(docs), "updated": res["updated"], "processed_total": processed}))
    print(json.dumps({"ok": True, "processed": processed, "db": DB_NAME}, indent=2))

if __name__ == "__main__":
    main()
