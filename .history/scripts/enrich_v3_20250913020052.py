#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
enrich_v3.py
------------
Windowed v3 enrichment with 43 JD-only fields, resume-safe, 5-min LLM timeout,
100-job windows (70/15/15 split), and per-job printing on success.

Place at: rover/scripts/enrich_v3.py
"""

import os, sys, re, json, html, platform, signal, threading, time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml
from pymongo import MongoClient, UpdateOne

# --------------------------- Paths / Config ---------------------------

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CFG_PATH = ROOT / "config" / "enrichment_v3.yaml"               # optional
PROMPT_MISSING_PATH = ROOT / "prompts" / "jd_v3_missing.prompt"  # optional
SCHEMA_PATH = ROOT / "schemas" / "jd_v3.schema.json"             # optional

# Ontology packs (optional; script has safe fallbacks)
SKILLS_TECH = ROOT / "config" / "ontology" / "skills_tech.yaml"
SKILLS_HR_TS = ROOT / "config" / "ontology" / "skills_hr_ts.yaml"
SKILLS_TRANSPORT = ROOT / "config" / "ontology" / "skills_transport.yaml"

# --------------------------- Env / DB ---------------------------

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

MONGO_URI = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
if not MONGO_URI:
    print("ERROR: Set MONGO_URI (or MONGODB_URI) in your .env")
    sys.exit(1)

DB_NAME = os.environ.get("JOBS_DB_NAME", "refjobs")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
jobs_col = db["jobs"]
cache_col = db.get_collection("jobs_llm_cache")  # { key, value, created_at }

# --------------------------- Optional deps ---------------------------

try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# --------------------------- Load Config (with safe defaults) ---------------------------

DEFAULT_CFG = {
    "version": 3,
    "model_version": "v3.0",
    "prompt_version": "v3.0",
    "selector": {"rule": "windowed_100"},
    "batch": {"size": 100, "max_concurrency": 2, "max_jobs_total": 0},
    "timeouts": {"llm": 300},
    "llm": {
        "base_url": "http://127.0.0.1:8080",
        "model": "Phi-3.5-mini-instruct-Q5_K_S.gguf",
        "temperature": 0.1,
        "max_input_chars": 3500,
        "max_output_tokens": 256
    },
    "qa": {"publish_min_score": 75},
    "ontology": {
        "remote_patterns": [r"\bfully remote\b", r"\bremote[- ]first\b", r"\bwork from home\b", r"\banywhere\b"],
        "hybrid_patterns": [r"\bhybrid\b", r"\b\d+ days onsite\b", r"\b\d{1,3}% onsite\b"],
        "onsite_patterns": [r"\bonsite\b", r"\bon[- ]site\b"]
    }
}

CFG = DEFAULT_CFG
if CFG_PATH.exists():
    try:
        user_cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}
        # shallow merge
        CFG.update(user_cfg)
    except Exception as e:
        print(f"[CFG WARN] Could not read {CFG_PATH}: {e}. Using defaults.")

LLM = CFG.get("llm", {})
LLAMA_BASE_URL = LLM.get("base_url", DEFAULT_CFG["llm"]["base_url"])
LLAMA_MODEL = LLM.get("model", DEFAULT_CFG["llm"]["model"])
TEMPERATURE = float(LLM.get("temperature", 0.1))
MAX_INPUT_CHARS = int(LLM.get("max_input_chars", 3500))
MAX_OUT_TOKENS = int(LLM.get("max_output_tokens", 256))
PROMPT_VERSION = CFG.get("prompt_version", "v3.0")
MODEL_VERSION = CFG.get("model_version", "v3.0")
LLM_TIMEOUT = int(CFG.get("timeouts", {}).get("llm", 300))

BATCH_SIZE = int(CFG.get("batch", {}).get("size", 100))
MAX_CONCURRENCY = int(CFG.get("batch", {}).get("max_concurrency", 2))
MAX_JOBS_TOTAL = int(CFG.get("batch", {}).get("max_jobs_total", 0))

# --------------------------- CLI ---------------------------

import argparse
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", dest="job_id", help="Process only this job _id")
    ap.add_argument("--retry-failures", action="store_true", help="Before assigning a new window, retry prior failures/timeouts")
    ap.add_argument("--window-size", type=int, default=BATCH_SIZE, help="Window size (default 100)")
    ap.add_argument("--ask-on-error", action="store_true", help="Ask to skip/retry on error (auto-skip after 5 min). Windows fallback = auto-skip.")
    ap.add_argument("--print", dest="print_all", action="store_true", help="Print full jd_v3 after each successful enrichment")
    ap.add_argument("--print-fields", dest="print_fields", help="Comma-separated subset of jd_v3 keys to print")
    ap.add_argument("--print-html", action="store_true", help="Also print CLEANED TEXT (from description_html)")
    ap.add_argument("--print-raw-html", action="store_true", help="Also print ORIGINAL HTML (truncated for readability)")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    return ap.parse_args()

ARGS = parse_args()
PRINT_ALL = bool(ARGS.print_all)
PRINT_FIELDS = [k.strip() for k in (ARGS.print_fields or "").split(",") if k.strip()] or None
PRINT_HTML = bool(ARGS.print_html)
PRINT_RAW_HTML = bool(ARGS.print_raw_html)
if PRINT_ALL or PRINT_FIELDS or PRINT_HTML or PRINT_RAW_HTML:
    ARGS.no_progress = True  # cleaner logs

# --------------------------- Utils ---------------------------

SESSION = requests.Session()

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def clean_html_text(html_str: Optional[str]) -> str:
    if not html_str: return ""
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html_str, "lxml")
            for s in soup(["script", "style"]): s.extract()
            text = soup.get_text("\n")
        except Exception:
            text = re.sub(r"<[^>]+>", " ", html_str)
    else:
        text = re.sub(r"<[^>]+>", " ", html_str)
    text = html.unescape(text)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def checksum_text(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))

# --------------------------- Sectioning & Parsers (rule-first) ---------------------------

# --- HEADER + SECTIONING (replace old HDR_RE, split_sections, bulleted) ---

HDR_RE = re.compile(
    r"^\s*(?:"
    r"(?:what\s+you(?:'|’)ll\s+do|what\s+you\s+will\s+do|key\s+responsibilities|responsibilities|the\s+impact\s+you(?:'|’)ll\s+have\s+here)"
    r"|(?:requirements|qualifications|what\s+we(?:'|’)re\s+looking\s+for|about\s+you)"
    r"|(?:benefits|perks|why\s+join(?:\s+us)?|why\s+[A-Za-z0-9_.-]+)"
    r"|(?:location)"
    r")\s*[:\-]?\s*$",
    re.I | re.M
)

def split_sections(text: str) -> Dict[str, str]:
    """
    Greedy but stable: walk line-by-line, open a new bucket on header, append
    lines until the next header. Keeps structure cleaner than free-form scan.
    """
    lines = [ln.rstrip() for ln in text.split("\n")]
    buckets: Dict[str, List[str]] = {}
    cur = "intro"
    buckets[cur] = []
    for ln in lines:
        if HDR_RE.match(ln.strip()):
            label = ln.strip().lower()
            if "responsib" in label or "what you" in label or "impact" in label or "key respons" in label:
                cur = "responsibilities"
            elif "requirement" in label or "qualification" in label or "what we" in label or "about you" in label:
                cur = "requirements"
            elif "benefit" in label or "perk" in label or "why join" in label or "why " in label:
                cur = "benefits"
            elif "location" in label:
                cur = "location"
            else:
                cur = re.sub(r"[^a-z]+", "_", label).strip("_") or "intro"
            buckets.setdefault(cur, [])
            continue
        buckets.setdefault(cur, []).append(ln)
    return {k: "\n".join(v).strip() for k, v in buckets.items()}

def bulleted(segment: str) -> List[str]:
    """
    Turn a section into clean bullets. Drop obvious headings/labels and empty crumbs.
    """
    if not segment:
        return []
    # Split by lines, keep only meaningful content (remove dangling labels like "Strategic Content Planning:")
    raw_lines = [x.strip(" -•\t·").strip() for x in segment.split("\n")]
    out = []
    for s in raw_lines:
        if not s or len(s) < 3:
            continue
        # Drop pure header-ish lines ending with ":" with no sentence following
        if re.match(r"^[A-Z][A-Za-z0-9 ()/&’'–-]{2,}:\s*$", s):
            continue
        out.append(s)
    # dedupe
    uniq, seen = [], set()
    for x in out:
        xl = re.sub(r"\s+", " ", x).strip().lower()
        if xl not in seen:
            seen.add(xl)
            uniq.append(x.strip())
    return uniq[:20]


SENIORITY_MAP = [
    (re.compile(r"\b(intern|working student)\b", re.I), "intern"),
    (re.compile(r"\b(junior|jr\.)\b", re.I), "junior"),
    (re.compile(r"\b(senior|sr\.)\b", re.I), "senior"),
    (re.compile(r"\b(principal|staff|lead)\b", re.I), "lead"),
    (re.compile(r"\b(manager|head|director)\b", re.I), "manager"),
]

def title_normalize(raw_title: str) -> Tuple[str, str]:
    if not raw_title: return ("", "")
    t = raw_title.strip().lower()
    seniority = ""
    for rx, label in SENIORITY_MAP:
        if rx.search(t): seniority = label; break
    t = re.sub(r"[^a-z0-9 ]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return (t.replace(" ", "_"), seniority)

REMOTE_PATTERNS = [re.compile(p, re.I) for p in CFG.get("ontology", {}).get("remote_patterns", [])]
HYBRID_PATTERNS = [re.compile(p, re.I) for p in CFG.get("ontology", {}).get("hybrid_patterns", [])]
ONSITE_PATTERNS = [re.compile(p, re.I) for p in CFG.get("ontology", {}).get("onsite_patterns", [])]

def derive_remote(text: str, provider: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    t = text.lower()
    mode, remote = "onsite", False
    if any(p.search(t) for p in REMOTE_PATTERNS): remote, mode = True, "remote"
    if any(p.search(t) for p in HYBRID_PATTERNS): remote, mode = True, "hybrid"
    if any(p.search(t) for p in ONSITE_PATTERNS): remote, mode = False, "onsite"
    regions = []
    loc = (provider or {}).get("candidate_required_location") or ""
    if loc: regions = [loc.strip()]
    return (remote, mode, regions)

SAL_CURRENCY_MAP = {"$":"USD", "€":"EUR", "£":"GBP"}
# --- SALARY (replace old SAL_RE, parse_salary) ---

SALARY_RANGE_RE = re.compile(
    r"(?P<prefix>(salary|compensation|pay|rate)[^\n:]*:\s*)?"
    r"(?P<cur>[$€£]|USD|EUR|GBP)?\s*"
    r"(?P<min>\d[\d,\.]*\s*[kK]?)\s*[-–—]\s*(?P<max>\d[\d,\.]*\s*[kK]?)"
    r"(?:\s*(?P<code>USD|EUR|GBP))?"
    r"(?:\s*(?:per|/)?\s*(?P<unit>hour|hr|day|month|mo|year|yr|annual))?",
    re.I
)

def _num_from(s: str) -> Optional[float]:
    try:
        s = s.lower().replace(",", "").strip()
        if s.endswith("k"):
            return float(s[:-1]) * 1000.0
        return float(s)
    except Exception:
        return None

def parse_salary(text: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    Find best salary candidate:
    - prefer ones with currency symbol/code and unit (/hr, /year)
    - penalize if nearby 'year(s)/yrs' refers to experience (false positive)
    - select highest score; return None if low confidence
    Handles things like "$4-$7 USD/hr".
    """
    t = text
    candidates = []
    for m in SALARY_RANGE_RE.finditer(t):
        span = m.span()
        window = t[max(0, span[0]-25):min(len(t), span[1]+25)].lower()

        # Penalize if clearly about experience years
        years_near = bool(re.search(r"\b(years?|yrs?)\b", window))
        # Boost if currency present
        has_cur = bool(m.group("cur") or m.group("code"))
        # Boost if unit present
        unit = (m.group("unit") or "").lower() or None
        has_unit = bool(unit)

        score = 0
        if has_cur: score += 3
        if has_unit: score += 2
        if years_near: score -= 5
        if m.group("prefix"): score += 2  # "Salary:" etc.

        mn = _num_from(m.group("min"))
        mx = _num_from(m.group("max"))
        cur = (m.group("cur") or m.group("code") or "").upper()
        if cur in {"$", "£", "€"}:
            cur = {"$":"USD","£":"GBP","€":"EUR"}[cur]
        if unit in {"yr"}: unit = "year"
        if unit in {"hr"}: unit = "hour"
        candidates.append({"score": score, "mn": mn, "mx": mx, "cur": cur or None, "unit": unit})

    if not candidates:
        return (None, None, None, None)
    best = max(candidates, key=lambda x: x["score"])
    if best["score"] <= 0:
        return (None, None, None, None)
    return (best["mn"], best["mx"], best["cur"], best["unit"])


LANG_MAP = {
    "english":"en","en":"en","eng":"en",
    "dutch":"nl","nederlands":"nl","nl":"nl",
    "german":"de","deutsch":"de","de":"de",
    "french":"fr","français":"fr","francais":"fr","fr":"fr",
    "italian":"it","italiano":"it","it":"it",
    "portuguese":"pt","português":"pt","portugues":"pt","pt":"pt",
    "spanish":"es","español":"es","espanol":"es","es":"es"
}
CEFR_LVLS = ["C2","C1","B2","B1","A2","A1"]

def parse_languages(text: str) -> List[Dict[str, Any]]:
    out = []
    t = text.lower()
    for lang_name, code in LANG_MAP.items():
        for lvl in CEFR_LVLS:
            if re.search(rf"\b{lang_name}\b.*\b{lvl}\b", t, re.I) or re.search(rf"\b{lvl}\b.*\b{lang_name}\b", t, re.I):
                out.append({"lang": code, "level": lvl, "source": "desc", "confidence": 0.8})
    for lang_name, code in LANG_MAP.items():
        if re.search(rf"\b{lang_name}\b", t, re.I):
            if not any(d["lang"]==code for d in out):
                out.append({"lang": code, "level": None, "source": "desc", "confidence": 0.5})
    return out[:6]

BENEFIT_VOCAB = {
    "remote":"Remote", "remote-first":"Remote", "work from home":"Remote",
    "pto":"PTO", "vacation":"PTO", "holiday":"PTO", "unlimited vacation":"PTO",
    "parental leave":"Parental Leave", "maternity":"Parental Leave", "paternity":"Parental Leave",
    "stock":"Equity", "equity":"Equity", "options":"Equity",
    "health":"Health", "medical":"Health", "dental":"Dental", "vision":"Vision",
    "retreat":"Retreat", "offsite":"Retreat", "company retreat":"Retreat",
    "sick pay":"Sick Pay", "sick leave":"Sick Pay",
    "flexible schedule":"Flexible Hours", "flexible hours":"Flexible Hours",
    "mentorship":"Mentorship",
    # transport-specific
    "per diem":"Per Diem", "mileage":"Mileage Reimbursement",
    "sign-on bonus":"Sign-on Bonus", "sign on bonus":"Sign-on Bonus", "bonus":"Bonus"
}

def normalize_benefits(raw_bullets: List[str]) -> List[str]:
    out = set()
    s = " | ".join(raw_bullets).lower()
    for k, v in BENEFIT_VOCAB.items():
        if k in s: out.add(v)
    return sorted(out)

def extract_role_category_and_specialty(title: str, text: str) -> Tuple[str, str]:
    t = f"{title} {text}".lower()

    # Marketing / Content
    if re.search(r"\b(copywriter|content marketer|content marketing|seo|social media|marketing)\b", t):
        return ("Marketing / Content", "B2B SaaS")

    # Trust & Safety
    if re.search(r"\b(trust\s*&?\s*safety|moderation|content policy)\b", t):
        return ("Trust & Safety", "Content Moderation")

    # Human Resources
    if re.search(r"\b(recruiter|talent\s+acquisition|sourcer|recruitment)\b", t):
        return ("Human Resources", "Talent Acquisition")

    # Infrastructure / DevOps / Platform
    if re.search(r"\b(devops|platform engineer|site reliability|sre|infrastructure engineer|infrastructure)\b", t):
        return ("Infrastructure / DevOps / Platform", "Platform/DevOps")

    # Software / Data Science
    if re.search(r"\b(data scientist|machine learning|ml)\b", t):
        return ("Software / Data", "Data Science")

    # Software / Dev
    if re.search(r"\b(software|backend|full[- ]stack|frontend|api|kubernetes|docker|node\.?js|typescript|javascript)\b", t):
        return ("Software / Dev", "Backend/Platform")

    # Logistics / Transport
    if re.search(r"\b(truck(ing)? driver|cdl|hgv|courier|delivery driver|last[- ]mile|otr|tanker|flatbed|box truck)\b", t):
        return ("Logistics / Transport", "Truck Driver")

    return ("Other", "")
fe
def extract_req_resp_benefits(sections: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    req = []; resp = []; ben = []
    for k, v in sections.items():
        if "requirement" in k or "qualification" in k or "about_you" in k:
            req.extend(bulleted(v))
        elif "responsibilit" in k or "what_you_will_do" in k or "what_you_ll_do" in k or "about_the_role" in k:
            resp.extend(bulleted(v))
        elif "benefit" in k or "perk" in k:
            ben.extend(bulleted(v))
    if not resp and sections.get("intro"):
        intro_bul = bulleted(sections["intro"])
        resp.extend(intro_bul[:6])
    uniq = lambda xs: list(dict.fromkeys(xs))
    return (uniq(req)[:20], uniq(resp)[:20], uniq(ben)[:20])

def dedupe_lists(a: List[str], b: List[str]) -> Tuple[List[str], List[str]]:
    aset = set(x.lower() for x in a)
    b2 = [x for x in b if x.lower() not in aset]
    return (a, b2)

def years_required(text: str) -> Optional[int]:
    m = re.search(r"(\d{1,2})\+?\s*(?:years|yrs)\b", text, re.I)
    if m:
        try: return int(m.group(1))
        except Exception: return None
    return None

def adjust_seniority(sen: str, yrs: Optional[int]) -> str:
    rank = ["intern","junior","mid","senior","lead","manager","director","vp","cxo"]
    if not sen and yrs is not None:
        if yrs >= 5: return "senior"
        if yrs >= 2: return "mid"
        return "junior"
    if sen in rank and yrs is not None:
        idx = rank.index(sen)
        if yrs >= 2 and idx < rank.index("mid"):
            return "mid"
    return sen

def load_yaml_list(p: Path) -> List[str]:
    if not p.exists(): return []
    y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    xs = y.get("skills", []) if isinstance(y, dict) else (y or [])
    return [str(s).strip().lower() for s in xs if str(s).strip()]

SKILLS_TECH_LIST = set(load_yaml_list(SKILLS_TECH))
SKILLS_HR_TS_LIST = set(load_yaml_list(SKILLS_HR_TS))
SKILLS_TRANSPORT_LIST = set(load_yaml_list(SKILLS_TRANSPORT))

def pick_allowlist(role_category: str) -> set:
    if role_category.startswith("Human Resources") or role_category.startswith("Trust & Safety"):
        return SKILLS_HR_TS_LIST
    if role_category.startswith("Logistics / Transport"):
        return SKILLS_TRANSPORT_LIST
    if role_category.startswith("Software"):
        return SKILLS_TECH_LIST
    return SKILLS_TECH_LIST  # safe default

def seed_skills(text: str, provider_tags: List[str], allow: set) -> List[str]:
    out = set()
    for t in provider_tags or []:
        tl = t.strip().lower()
        if tl in allow: out.add(tl)
    for sk in allow:
        if re.search(rf"\b{re.escape(sk)}\b", text, re.I):
            out.add(sk)
    return sorted(out)

def keywords_taxonomy(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-\+]{2,}", text.lower())
    freq: Dict[str,int] = {}
    stop = set("the and for with you your our this that from will are have has can into to of in on a an as by be is".split())
    for w in words:
        if w in stop: continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:25]
    return [w for w, _ in ranked]

# Transport-specific
def parse_trucking_certs(text: str) -> List[str]:
    out = set(); t = text.lower()
    if re.search(r"\bcdl\b", t): out.add("CDL")
    if re.search(r"\bcdl\s*(class|cls)\s*a\b", t): out.add("CDL Class A")
    if re.search(r"\bcdl\s*(class|cls)\s*b\b", t): out.add("CDL Class B")
    if re.search(r"\bhazmat\b|\badr\b", t): out.add("HazMat/ADR")
    if re.search(r"\btanker\b", t): out.add("Tanker")
    if re.search(r"\bdoubles?\b|\btriples?\b", t): out.add("Doubles/Triples")
    if re.search(r"\bcpc\b", t): out.add("CPC")
    if re.search(r"\bhgv\b", t): out.add("HGV")
    return sorted(out)

def parse_trucking_compliance(text: str) -> List[str]:
    hits = set(); t = text.lower()
    if "dot" in t: hits.add("DOT")
    if "fmcsa" in t: hits.add("FMCSA")
    if re.search(r"\bhos\b|hours[- ]of[- ]service", t): hits.add("HOS")
    if re.search(r"\beld\b", t): hits.add("ELD")
    if re.search(r"\btacho(graph)?\b", t): hits.add("Tachograph")
    return sorted(hits)

def parse_trucking_schedule(text: str) -> Tuple[Optional[str], Optional[str]]:
    t = text.lower()
    ws = []; note = None
    if re.search(r"\botr\b|over[- ]the[- ]road", t): ws.append("OTR")
    if re.search(r"\bregional\b", t): ws.append("Regional")
    if re.search(r"\blocal\b|home\s+daily", t): ws.append("Local")
    if re.search(r"\bnight(s)? shift\b|overnight", t): ws.append("Night Shift")
    if re.search(r"\bweekend(s)?\b", t): ws.append("Weekend Shift")
    m = re.search(r"home (daily|weekly|biweekly)", t)
    if m: note = f"home {m.group(1)}"
    return (", ".join(sorted(set(ws))) if ws else None, note)

# --------------------------- LLM (json-only) ---------------------------

def load_missing_prompt() -> str:
    if PROMPT_MISSING_PATH.exists():
        try:
            return PROMPT_MISSING_PATH.read_text(encoding="utf-8")
        except Exception:
            pass
    # fallback
    return ("Extract ONLY the requested keys from the job text.\n"
            "- Use EXACT wording from the text when possible.\n"
            "- If a field is not explicitly stated, return null (scalars) or [] (arrays).\n"
            "- Keep arrays concise (max 8 items; outcomes up to 10).\n"
            "- Do not invent salaries, dates, or locations.\n"
            "- JSON only, no markdown.\n"
            "Key semantics:\n"
            "- responsibilitiesList: concrete actions the role performs (verbs).\n"
            "- requirements: qualifications/skills/experience needed to get hired.\n"
            "- workHoursWindow: stated time-zone overlap/hours expectation.\n"
            "- outcomesKpis: measurable results/success metrics.\n"
            "- summary_tldr: one sentence (<= 240 chars) plain English role summary.\n")

def llm_chat_json(text: str, keys_needed: List[str]) -> Optional[Dict[str, Any]]:
    sys_prompt = ("You extract structured fields from job descriptions. "
                  "Return STRICT JSON with ONLY the requested keys. No prose, no code fences.")
    user = (
        f"Return JSON with EXACTLY these keys (and only these keys): {json.dumps(keys_needed)}.\n"
        "If a field is not explicitly present in the text, output null or [] accordingly.\n\n"
        f"TEXT:\n{text.strip()[:MAX_INPUT_CHARS]}"
    )
    payload = {
        "model": LLM["model"],
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": load_missing_prompt() + "\n\n---\n" + user}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_OUT_TOKENS,
        "response_format": {"type": "json_object"}
    }
    try:
        r = SESSION.post(f"{LLAMA_BASE_URL.rstrip('/')}/v1/chat/completions", json=payload, timeout=LLM_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        content = (data["choices"][0]["message"]["content"] or "").strip()
        try:
            return json.loads(content)
        except Exception:
            m = re.search(r"\{.*\}", content, flags=re.DOTALL)
            if m: return json.loads(m.group(0))
            return None
    except Exception as e:
        print(f"[LLM ERROR] {e}")
        return None

def cache_get(key: str) -> Optional[Dict[str, Any]]:
    doc = cache_col.find_one({"key": key}, {"_id": 0, "value": 1})
    return doc.get("value") if doc else None

def cache_put(key: str, value: Dict[str, Any]) -> None:
    try:
        cache_col.update_one({"key": key}, {"$set": {"value": value, "created_at": now_iso()}}, upsert=True)
    except Exception as e:
        print(f"[CACHE WARN] {e}")

# --------------------------- QA ---------------------------

def qa_flags_and_score(jd: Dict[str, Any]) -> Tuple[List[str], int]:
    flags = []
    if not jd.get("normalizedTitle"): flags.append("MISSING_TITLE")
    if not jd.get("seniorityLevel"): flags.append("MISSING_SENIORITY")
    if len(jd.get("responsibilitiesList") or []) < 3: flags.append("RESP_TOO_SHORT")
    yrs = jd.get("yearsExperienceRequired"); sen = jd.get("seniorityLevel")
    if yrs is not None and sen in ("intern","junior") and yrs >= 2: flags.append("SENIORITY_INCONSISTENT")
    rm = jd.get("remoteMode")
    if rm not in (None, "remote", "hybrid", "onsite"): flags.append("REMOTE_ENUM_INVALID")
    ct = jd.get("compensationText", ""); cp = jd.get("compensationParsed")
    if ct and not (cp and (cp.get("min") or cp.get("max"))): flags.append("SALARY_PARSE_FAIL")
    if jd.get("benefitsRaw") and not jd.get("benefitsNormalized"): flags.append("BENEFITS_NOT_NORMALIZED")
    if len(jd.get("toolsTechnologies") or []) < 2 and (jd.get("roleCategory") or "").startswith("Software"):
        flags.append("SKILLS_TOO_FEW")
    score = 100
    penalties = {
        "MISSING_TITLE":15, "MISSING_SENIORITY":10, "RESP_TOO_SHORT":10,
        "SENIORITY_INCONSISTENT":10, "REMOTE_ENUM_INVALID":10,
        "SALARY_PARSE_FAIL":10, "BENEFITS_NOT_NORMALIZED":5, "SKILLS_TOO_FEW":10
    }
    for fl in flags: score -= penalties.get(fl, 5)
    return (flags, max(0, min(100, score)))

# --------------------------- 43-field skeleton ---------------------------

def jd_empty_43() -> Dict[str, Any]:
    return {
        "normalizedTitle": None,
        "titleAliases": [],
        "roleCategory": None,
        "roleSpecialty": None,
        "seniorityLevel": None,
        "employmentType": None,
        "workSchedule": None,
        "workHoursWindow": None,
        "remoteMode": None,
        "remoteEligibilityRegions": [],
        "locationMentions": [],
        "relocationOffered": None,
        "travelRequirementPct": None,
        "visaSponsorshipStatement": None,
        "workAuthorizationRequirements": [],
        "languageRequirements": [],
        "yearsExperienceRequired": None,
        "educationRequired": {"level": None, "required": None, "preferred": None},
        "certificationsRequired": [],
        "securityOrClearance": [],
        "responsibilitiesList": [],
        "outcomesKpis": [],
        "teamStructureSignals": [],
        "managementScope": None,
        "toolsTechnologies": [],
        "methodsFrameworks": [],
        "industryDomain": [],
        "productArea": [],
        "dataSensitivity": [],
        "complianceRegulations": [],
        "benefitsRaw": [],
        "benefitsNormalized": [],
        "compensationText": "",
        "compensationParsed": {"min": None, "max": None, "currencySymbol": None, "period": None},
        "perksExtras": [],
        "cultureSignals": [],
        "deiStatementPresence": None,
        "equipmentOrAllowances": [],
        "applicationProcess": [],
        "startDateUrgency": None,
        "contractDuration": None,
        "workEnvironment": [],
        "keywordsTaxonomy": []
    }

# --------------------------- Enrichment Core ---------------------------

def enrich_one(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a dict with:
      - jd_v3 (43 fields)
      - jd_v3_meta
      - jd_v3_qa
      - legacy convenience fields
      - cleaned_text
      - raw_html_trunc
    Raises on hard errors; caller handles marking status.
    """
    raw = job.get("raw") or {}
    desc_html = job.get("description_html") or ""
    cleaned = clean_html_text(desc_html)
    if not cleaned:
        raise ValueError("Empty description text")

    text_ck = checksum_text(cleaned)

    jd = jd_empty_43()

    # core title/seniority
    nt, sen = title_normalize(job.get("title") or "")
    jd["normalizedTitle"] = nt or None
    jd["titleAliases"] = [job.get("title")] if job.get("title") else []
    jd["seniorityLevel"] = sen or None

    # role category/specialty
    role_cat, role_spec = extract_role_category_and_specialty(job.get("title") or "", cleaned)
    jd["roleCategory"] = role_cat
    jd["roleSpecialty"] = role_spec or None

    # employment type (provider)
    emp = (job.get("employment_type") or "").lower() or (raw.get("job_type") or "").lower()
    jd["employmentType"] = emp if emp in ("full_time","part_time","contract","internship","temporary") else None

    # remote
    _, remote_mode, regions = derive_remote(cleaned, raw)
    jd["remoteMode"] = remote_mode
    jd["remoteEligibilityRegions"] = regions

    # location mentions (simple)
    loc_mentions = []
    for m in re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", cleaned):
        if m.lower() in ("we","our","you","the","and","or"): continue
        if len(m) >= 3: loc_mentions.append(m)
    jd["locationMentions"] = sorted(list(set(loc_mentions)))[:10]

    # sections
    sections = split_sections(cleaned)
    req, resp, ben = extract_req_resp_benefits(sections)
    req, resp = dedupe_lists(req, resp)
    jd["responsibilitiesList"] = resp
    jd["benefitsRaw"] = ben
    jd["benefitsNormalized"] = normalize_benefits(ben)

    # years/seniority
    yrs = years_required(cleaned)
    jd["yearsExperienceRequired"] = yrs
    jd["seniorityLevel"] = adjust_seniority(jd["seniorityLevel"] or "", yrs) or jd["seniorityLevel"]

    # salary
    mn, mx, cur, per = parse_salary(cleaned)
    if any([mn, mx, cur, per]):
        m = SAL_RE.search(cleaned)
        jd["compensationText"] = m.group(0) if m else ""
        jd["compensationParsed"] = {"min": mn, "max": mx, "currencySymbol": cur, "period": per}

    # languages
    jd["languageRequirements"] = parse_languages(cleaned)

    # skills & methods
    allow = pick_allowlist(role_cat)
    seeded = seed_skills(cleaned, (raw.get("tags") or job.get("tags") or []), allow)
    jd["toolsTechnologies"] = seeded
    meth = []
    if re.search(r"\b(agile|scrum|kanban)\b", cleaned, re.I): meth.append("agile")
    if re.search(r"\b(tdd|testing)\b", cleaned, re.I): meth.append("testing")
    if re.search(r"\b(mlops)\b", cleaned, re.I): meth.append("mlops")
    jd["methodsFrameworks"] = sorted(list(set(meth)))

    # domain/product area
    jd["industryDomain"] = list(sorted(set([x for x in [
        role_cat.split(" / ")[-1],
        "recruitment services" if "recruit" in cleaned.lower() else "",
        "renewable energy" if "energy" in cleaned.lower() else ""
    ] if x])))
    jd["productArea"] = []

    # compliance/sensitivity
    if re.search(r"\b(gdpr|hipaa|soc2|iso)\b", cleaned, re.I):
        jd.setdefault("complianceRegulations", []).extend(sorted(set(re.findall(r"\b(gdpr|hipaa|soc2|iso)\b", cleaned, re.I))))
    if re.search(r"\b(moderation|trust & safety|trust and safety|content policy)\b", cleaned, re.I):
        jd["dataSensitivity"] = ["content moderation"]

    # perks/culture/process
    if "retreat" in cleaned.lower(): jd.setdefault("perksExtras", []).append("retreat")
    jd["cultureSignals"] = []
    if "founder-led" in cleaned.lower(): jd["cultureSignals"].append("founder-led")
    if "remote-first" in cleaned.lower(): jd["cultureSignals"].append("remote-first")
    jd["cultureSignals"] = sorted(list(set(jd["cultureSignals"])))
    jd["deiStatementPresence"] = True if re.search(r"equal opportunity|diversity|all backgrounds", cleaned, re.I) else False
    jd["applicationProcess"] = ["apply via platform"] if re.search(r"\bapply\b", cleaned, re.I) else []
    jd["workEnvironment"] = ["remote"] if remote_mode in ("remote","hybrid") else []
    jd["keywordsTaxonomy"] = keywords_taxonomy(cleaned)

    # Transport extras
    if role_cat.startswith("Logistics / Transport"):
        certs = parse_trucking_certs(cleaned)
        if certs:
            jd["certificationsRequired"] = sorted(list(set((jd.get("certificationsRequired") or []) + certs)))
        comp = parse_trucking_compliance(cleaned)
        if comp:
            jd["complianceRegulations"] = sorted(list(set((jd.get("complianceRegulations") or []) + comp)))
        ws, note = parse_trucking_schedule(cleaned)
        if ws: jd["workSchedule"] = ws
        if note: jd["workEnvironment"] = sorted(list(set((jd.get("workEnvironment") or []) + [note])))

    # ---------------- LLM micro-call (only if needed) ----------------
    need_keys = []
    if len(jd["responsibilitiesList"]) < 3: need_keys.append("responsibilitiesList")
    if len(req) < 3: need_keys.append("requirements")
    if not jd.get("workHoursWindow") and re.search(r"\b(overlap|CET|GMT|PST|EST)\b", cleaned): need_keys.append("workHoursWindow")
    if not jd.get("outcomesKpis"): need_keys.append("outcomesKpis")
    need_keys.append("summary_tldr")

    llm_added = {}
    if need_keys:
        cache_key = f"{LLM['model']}:{PROMPT_VERSION}:{text_ck}:{','.join(sorted(need_keys))}"
        cached = cache_get(cache_key)
        if cached:
            llm_added = cached
        else:
            llm_added = llm_chat_json(cleaned, need_keys) or {}
            if llm_added: cache_put(cache_key, llm_added)

    if isinstance(llm_added.get("responsibilitiesList"), list) and len(jd["responsibilitiesList"]) < 3:
        jd["responsibilitiesList"] = dedupe_lists(jd["responsibilitiesList"], llm_added["responsibilitiesList"])[0]
    if isinstance(llm_added.get("requirements"), list) and len(req) < 3:
        req = llm_added["requirements"][:20]
    if isinstance(llm_added.get("workHoursWindow"), str) and not jd.get("workHoursWindow"):
        jd["workHoursWindow"] = llm_added["workHoursWindow"][:80]
    if isinstance(llm_added.get("outcomesKpis"), list) and not jd.get("outcomesKpis"):
        jd["outcomesKpis"] = llm_added["outcomesKpis"][:10]
    summary = (llm_added.get("summary_tldr") or "").strip()

    jd["requirementsList"] = req

    # QA
    flags, score = qa_flags_and_score(jd)
    meta = {
        "modelVersion": MODEL_VERSION,
        "promptVersion": PROMPT_VERSION,
        "textChecksum": text_ck,
        "enrichedAt": now_iso(),
        "status": "success",
        "attempts": 1
    }
    legacy = {
        "normalized_title": jd["normalizedTitle"],
        "seniority": jd["seniorityLevel"],
        "summary_tldr": summary[:240] if summary else "",
        "remote_mode": jd["remoteMode"],
        "quality_score": score
    }

    raw_html_trunc = (job.get("description_html") or "")
    if len(raw_html_trunc) > 2000:  # keep logs readable
        raw_html_trunc = raw_html_trunc[:2000] + "\n...[truncated]..."

    return {
        "jd_v3": jd,
        "jd_v3_meta": meta,
        "jd_v3_qa": {"score": score, "flags": flags},
        "legacy": legacy,
        "cleaned_text": cleaned,
        "raw_html_trunc": raw_html_trunc
    }

# --------------------------- Windowing & Selection ---------------------------

def next_window_number() -> int:
    doc = jobs_col.find_one(
        {"jd_v3_eval_window": {"$exists": True}},
        sort=[("jd_v3_eval_window", -1)],
        projection={"jd_v3_eval_window": 1, "_id": 0}
    )
    return int(doc["jd_v3_eval_window"]) + 1 if doc else 1

def assign_window(window_size: int) -> Tuple[int, int]:
    """
    Assigns next window to the next N unassigned jobs.
    Returns (window_number, assigned_count). If 0 assigned, returns (wn, 0).
    """
    wn = next_window_number()
    cur = jobs_col.find(
        {"jd_v3_eval_window": {"$exists": False}},
        projection={"_id":1, "last_seen_at":1, "apply_url":1},
        sort=[("last_seen_at", 1), ("_id", 1)],
        limit=window_size
    )
    jobs = list(cur)
    if not jobs:
        return (wn, 0)
    ops = []
    for idx, j in enumerate(jobs):
        if idx >= window_size: break
        split = "train" if idx < 70 else ("val" if idx < 85 else "test")
        ops.append(UpdateOne(
            {"_id": j["_id"]},
            {"$set": {"jd_v3_eval_window": wn, "jd_v3_eval_index": idx, "jd_v3_eval_split": split}}
        ))
    if ops:
        jobs_col.bulk_write(ops, ordered=False)
    return (wn, len(ops))

def find_jobs_for_window(wn: int) -> List[Dict[str, Any]]:
    q = {"jd_v3_eval_window": wn}
    proj = {
        "_id":1, "title":1, "company":1, "source":1, "raw":1, "employment_type":1,
        "description_html":1, "tags":1,
        "last_seen_at":1, "jd_v3_meta":1, "jd_v3_eval_index":1, "jd_v3_eval_split":1
    }
    return list(jobs_col.find(q, projection=proj, sort=[("jd_v3_eval_index", 1)]))

def find_prior_failures() -> List[Dict[str, Any]]:
    q = {"jd_v3_meta.status": {"$in": ["error","timeout"]}}
    proj = {"_id":1, "title":1, "raw":1, "employment_type":1, "description_html":1, "tags":1, "jd_v3_meta":1}
    return list(jobs_col.find(q, projection=proj, sort=[("jd_v3_meta.enrichedAt", 1)], limit=200))

# --------------------------- Progress UI ---------------------------

def render_progress(done: int, total: int, width: int = 28) -> str:
    if total <= 0: return f"{done} / ?"
    frac = min(1.0, done/total)
    filled = int(round(frac * width))
    bar = "█" * filled + " " * (width - filled)
    return f"{bar}  {done:,} / {total:,} ({frac*100:4.1f}%)"

# --------------------------- Ask-on-error (with Windows fallback) ---------------------------

def timed_input(prompt: str, timeout: int = 300) -> Optional[str]:
    """Returns input or None on timeout. On Windows, falls back to None immediately."""
    if platform.system().lower().startswith("win"):
        print("[ASK] Windows timed input not supported; auto-skip.")
        return None
    result = [None]
    def _get():
        try:
            result[0] = input(prompt)
        except Exception:
            result[0] = None
    th = threading.Thread(target=_get, daemon=True)
    th.start()
    th.join(timeout)
    return result[0]

# --------------------------- Printing per job ---------------------------

def print_success_artifacts(job: Dict[str, Any], enriched: Dict[str, Any]) -> None:
    _id = job.get("_id")
    title = job.get("title") or ""
    comp = (job.get("company") or {}).get("name") or ""
    print("\n" + "="*80)
    print(f"JOB {_id} — {title} @ {comp}")
    print("="*80)
    if PRINT_RAW_HTML:
        print("\n--- ORIGINAL HTML (truncated) ---")
        print(enriched["raw_html_trunc"])
    if PRINT_HTML:
        print("\n--- CLEANED TEXT ---")
        print(enriched["cleaned_text"])
    # Enriched fields (labels + values)
    jd = enriched["jd_v3"]
    print("\n--- ENRICHED FIELDS (jd_v3) ---")
    def p(label, val):
        if isinstance(val, (list, tuple)):
            print(f"{label}: " + ("; ".join([json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x) for x in val]) if val else "[]"))
        elif isinstance(val, dict):
            print(f"{label}: {json.dumps(val, ensure_ascii=False)}")
        else:
            print(f"{label}: {val if val not in (None, '') else 'null'}")
    # Print all or subset
    if PRINT_FIELDS:
        for key in PRINT_FIELDS:
            p(key, jd.get(key))
    else:
        for key in jd.keys():
            p(key, jd.get(key))
    print("-"*80)

# --------------------------- Main processing ---------------------------

def process_jobs(docs: List[Dict[str, Any]], totals: Dict[str, int]) -> None:
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        fut = {ex.submit(enrich_one, d): d for d in docs}
        for f in as_completed(fut):
            job = fut[f]
            try:
                out = f.result()
                # write success
                set_fields = {
                    "jd_v3": out["jd_v3"],
                    "jd_v3_meta": out["jd_v3_meta"],
                    "jd_v3_qa": out["jd_v3_qa"],
                    # legacy mirrors
                    "normalized_title": out["legacy"]["normalized_title"],
                    "seniority": out["legacy"]["seniority"],
                    "summary_tldr": out["legacy"]["summary_tldr"],
                    "remote_mode": out["legacy"]["remote_mode"],
                    "quality_score": out["legacy"]["quality_score"],
                }
                jobs_col.update_one({"_id": job["_id"]}, {"$set": set_fields}, upsert=False)
                totals["success"] += 1
                # prints
                if PRINT_ALL or PRINT_FIELDS or PRINT_HTML or PRINT_RAW_HTML:
                    print_success_artifacts(job, out)
            except Exception as e:
                # error handling: mark and maybe ask
                msg = str(e)
                status = "error"
                if "timed out" in msg.lower() or "timeout" in msg.lower():
                    status = "timeout"
                jobs_col.update_one({"_id": job["_id"]}, {"$set": {
                    "jd_v3_meta.status": status,
                    "jd_v3_meta.last_error": msg,
                    "jd_v3_meta.enrichedAt": now_iso()
                }})
                totals[status] += 1
                if ARGS.ask_on_error:
                    print(f"\n[ERROR] Job {job.get('_id')}: {msg}")
                    ans = timed_input("(s)kip / (r)etry once / (a)lways skip this run [default: skip]: ", timeout=300)
                    if ans and ans.strip().lower().startswith("r"):
                        try:
                            out2 = enrich_one(job)
                            set_fields = {
                                "jd_v3": out2["jd_v3"],
                                "jd_v3_meta": out2["jd_v3_meta"],
                                "jd_v3_qa": out2["jd_v3_qa"],
                                "normalized_title": out2["legacy"]["normalized_title"],
                                "seniority": out2["legacy"]["seniority"],
                                "summary_tldr": out2["legacy"]["summary_tldr"],
                                "remote_mode": out2["legacy"]["remote_mode"],
                                "quality_score": out2["legacy"]["quality_score"],
                            }
                            jobs_col.update_one({"_id": job["_id"]}, {"$set": set_fields}, upsert=False)
                            totals["success"] += 1
                            totals[status] -= 1
                            if PRINT_ALL or PRINT_FIELDS or PRINT_HTML or PRINT_RAW_HTML:
                                print_success_artifacts(job, out2)
                        except Exception as e2:
                            jobs_col.update_one({"_id": job["_id"]}, {"$set": {
                                "jd_v3_meta.status": "error",
                                "jd_v3_meta.last_error": str(e2),
                                "jd_v3_meta.enrichedAt": now_iso()
                            }})
                            # leave counters as-is

# --------------------------- Run ---------------------------

def main():
    # optional: retry prior failures
    totals = {"success":0, "error":0, "timeout":0}
    if ARGS.retry_failures:
        prev = find_prior_failures()
        if prev:
            total_prev = len(prev)
            processed_prev = 0
            for chunk_start in range(0, total_prev, BATCH_SIZE):
                chunk = prev[chunk_start:chunk_start+BATCH_SIZE]
                process_jobs(chunk, totals)
                processed_prev += len(chunk)
                if not ARGS.no_progress:
                    sys.stdout.write("\r[Retry failures] " + render_progress(processed_prev, total_prev))
                    sys.stdout.flush()
            if not ARGS.no_progress:
                sys.stdout.write("\n")

    # If a specific job id is supplied, process only that
    if ARGS.job_id:
        doc = jobs_col.find_one({"_id": ARGS.job_id})
        if not doc:
            print(json.dumps({"ok": False, "error": "job not found", "id": ARGS.job_id}, indent=2))
            return
        process_jobs([doc], totals)
        print(json.dumps({"ok": True, "processed": 1, "window_assigned": 0, "success": totals["success"],
                          "error": totals["error"], "timeout": totals["timeout"]}, indent=2))
        return

    # assign next window of 100 (or window-size) if unassigned remain
    wn, assigned = assign_window(ARGS.window_size)
    if assigned == 0:
        # nothing to assign; nothing to do
        print(json.dumps({"ok": True, "processed": 0, "window_assigned": 0, "message": "No unassigned jobs found."}, indent=2))
        return

    # fetch jobs in this window
    docs = find_jobs_for_window(wn)
    total = len(docs)
    processed = 0

    # process in manageable chunks
    for i in range(0, total, BATCH_SIZE):
        chunk = docs[i:i+BATCH_SIZE]
        process_jobs(chunk, totals)
        processed += len(chunk)
        if not ARGS.no_progress:
            sys.stdout.write("\r[Window %d] %s  (succ:%d err:%d to:%d)" %
                             (wn, render_progress(processed, total), totals["success"], totals["error"], totals["timeout"]))
            sys.stdout.flush()

    if not ARGS.no_progress:
        sys.stdout.write("\n")

    # window split report
    # recompute per-split success average QA quickly
    agg = list(jobs_col.aggregate([
        {"$match": {"jd_v3_eval_window": wn, "jd_v3_meta.status": "success"}},
        {"$group": {"_id": "$jd_v3_eval_split", "avg_score": {"$avg": "$jd_v3_qa.score"}, "count": {"$sum": 1}}}
    ]))
    report = {a["_id"]: {"avg_score": round(float(a["avg_score"]), 2), "count": a["count"]} for a in agg}
    print(json.dumps({
        "ok": True,
        "window": wn,
        "assigned": assigned,
        "processed": processed,
        "success": totals["success"],
        "error": totals["error"],
        "timeout": totals["timeout"],
        "split_report": report
    }, indent=2))

if __name__ == "__main__":
    main()
