#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
v3 Job Enrichment Pipeline — rule-first, LLM-last, cacheable, QA-gated.
- Extracts 43 JD-only fields (see schemas/jd_v3.schema.json).
- Uses deterministic parsers first; calls LLM only for missing/ambiguous bits.
- Caches LLM outputs by (model, prompt_version, text_checksum).
- Writes to jobs collection:
    - jd_v3: { ... 43 fields ..., meta, qa }
    - top-level convenience fields for backward-compat.
"""

import os, sys, re, json, html, math
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import yaml
from pymongo import MongoClient, UpdateOne

# ---------- Optional deps ----------
try:
    from bs4 import BeautifulSoup
except Exception:
    BeautifulSoup = None

# ---------- Paths ----------
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CFG_PATH = ROOT / "config" / "enrichment_v3.yaml"
PROMPT_MISSING_PATH = ROOT / "prompts" / "jd_v3_missing.prompt"
SCHEMA_PATH = ROOT / "schemas" / "jd_v3.schema.json"
SKILLS_TECH = ROOT / "config" / "ontology" / "skills_tech.yaml"
SKILLS_HR_TS = ROOT / "config" / "ontology" / "skills_hr_ts.yaml"
SKILLS_TRANSPORT = ROOT / "config" / "ontology" / "skills_transport.yaml"
SKILLS_TRANSPORT_LIST = set(load_yaml_list(SKILLS_TRANSPORT))


# ---------- Env ----------
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
cache_col = db.get_collection("jobs_llm_cache")  # {key: str, value: dict, created_at}

# ---------- Load config ----------
if not CFG_PATH.exists():
    print(f"ERROR: config missing: {CFG_PATH}")
    sys.exit(1)
CFG = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}

LLM = CFG.get("llm", {})
LLAMA_BASE_URL = LLM.get("base_url", "http://127.0.0.1:8080")
LLAMA_MODEL = LLM.get("model", "Phi-3.5-mini-instruct-Q5_K_S.gguf")
TEMPERATURE = float(LLM.get("temperature", 0.1))
MAX_INPUT_CHARS = int(LLM.get("max_input_chars", 3500))
MAX_OUT_TOKENS = int(LLM.get("max_output_tokens", 256))
PROMPT_VERSION = CFG.get("prompt_version", "v3.0")
MODEL_VERSION = CFG.get("model_version", "v3.0")

BATCH = CFG.get("batch", {})
BATCH_SIZE = int(BATCH.get("size", 50))
MAX_CONCURRENCY = int(BATCH.get("max_concurrency", 2))
MAX_JOBS_TOTAL = int(BATCH.get("max_jobs_total", 0))

TIMEOUTS = CFG.get("timeouts", {})
LLM_TIMEOUT = int(TIMEOUTS.get("llm", 180))

QA_CFG = CFG.get("qa", {})
PUBLISH_MIN_SCORE = int(QA_CFG.get("publish_min_score", 75))

# Ontology
def load_yaml_list(p: Path) -> List[str]:
    if not p.exists(): return []
    y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    xs = y.get("skills", []) if isinstance(y, dict) else (y or [])
    return [str(s).strip().lower() for s in xs if str(s).strip()]

SKILLS_TECH_LIST = set(load_yaml_list(SKILLS_TECH))
SKILLS_HR_TS_LIST = set(load_yaml_list(SKILLS_HR_TS))

# ---------- Utils ----------
def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def clean_html(html_str: Optional[str]) -> str:
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
    # keep bullets and headings minimal structure
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def checksum_text(s: str) -> str:
    import hashlib
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

def clamp(n: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, n))

# ---------- Sectioning ----------
HDR_RE = re.compile(r"^\s*(requirements|qualifications|responsibilities|what you will do|what you’ll do|benefits|perks|about you|about the role)\s*[:\-]?\s*$", re.I|re.M)

def split_sections(text: str) -> Dict[str, str]:
    lines = [ln.strip() for ln in text.split("\n")]
    buckets: Dict[str, List[str]] = {}
    cur = "intro"
    for ln in lines:
        if re.match(HDR_RE, ln or ""):
            cur = re.sub(r"[^a-z]+", "_", ln.lower()).strip("_")
            buckets.setdefault(cur, [])
            continue
        buckets.setdefault(cur, []).append(ln)
    return {k: "\n".join(v).strip() for k, v in buckets.items()}

def bulleted(segment: str) -> List[str]:
    if not segment: return []
    out = []
    for raw in segment.split("\n"):
        s = raw.strip(" -•\t·").strip()
        if len(s) >= 3:
            out.append(s)
    # compact
    uniq, seen = [], set()
    for x in out:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl); uniq.append(x)
    return uniq[:20]

# ---------- Deterministic extractors ----------
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
        if rx.search(t):
            seniority = label; break
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
SAL_RE = re.compile(
    r"(?P<cur>[$€£])?\s*(?P<min>[0-9][0-9,\.kK]*)\s*[-–—]\s*(?P<max>[0-9][0-9,\.kK]*)\s*(?P<p>/\s*(year|yr|annual|month|mo|hour|hr))?",
    re.I
)

def parse_salary(text: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str], bool]:
    m = SAL_RE.search(text)
    if not m: return (None, None, None, None, False)
    cur_sym = (m.group("cur") or "").strip()
    currency = SAL_CURRENCY_MAP.get(cur_sym) if cur_sym else None
    def to_num(s: str) -> float:
        s = s.lower().replace(",", "").strip()
        if s.endswith("k"): return float(s[:-1]) * 1000.0
        return float(s)
    try:
        mn = to_num(m.group("min"))
        mx = to_num(m.group("max"))
    except Exception:
        return (None, None, currency, None, False)
    p = (m.group("p") or "").lower()
    period = None
    if "year" in p or "annual" in p or "yr" in p: period = "year"
    elif "month" in p or "mo" in p: period = "month"
    elif "hour" in p or "hr" in p: period = "hour"
    return (mn, mx, currency, period, False)

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
    # e.g., "Dutch C1", "English B2"
    for lang_name, code in LANG_MAP.items():
        for lvl in CEFR_LVLS:
            if re.search(rf"\b{lang_name}\b.*\b{lvl}\b", t, re.I) or re.search(rf"\b{lvl}\b.*\b{lang_name}\b", t, re.I):
                out.append({"lang": code, "level": lvl, "source": "desc", "confidence": 0.8})
    # if not CEFR, still capture plain language requirements (no level)
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
    "mentorship":"Mentorship"
}
def normalize_benefits(raw_bullets: List[str]) -> List[str]:
    out = set()
    s = " | ".join(raw_bullets).lower()
    for k, v in BENEFIT_VOCAB.items():
        if k in s:
            out.add(v)
    return sorted(out)

def extract_role_category_and_specialty(title: str, text: str) -> Tuple[str, str]:
    t = f"{title} {text}".lower()
    if re.search(r"\b(data scientist|machine learning|ml)\b", t): return ("Software / Data", "Data Science")
    if re.search(r"\b(software|backend|devops|kubernetes|docker|api)\b", t): return ("Software / Dev", "Backend/Platform")
    if re.search(r"\b(recruiter|talent|acquisition|sourcing)\b", t): return ("Human Resources", "Talent Acquisition")
    if re.search(r"\b(trust\s*&?\s*safety|moderation|content policy)\b", t): return ("Trust & Safety", "Content Moderation")
    return ("Other", "")

def extract_req_resp_benefits(sections: Dict[str, str]) -> Tuple[List[str], List[str], List[str]]:
    req = []
    resp = []
    ben = []
    # headings
    for k, v in sections.items():
        if "requirement" in k or "qualification" in k or "about_you" in k:
            req.extend(bulleted(v))
        elif "responsibilit" in k or "what_you_will_do" in k or "what_you_ll_do" in k or "about_the_role" in k:
            resp.extend(bulleted(v))
        elif "benefit" in k or "perk" in k:
            ben.extend(bulleted(v))
    # fallback: if one of req/resp empty, try intro for bullets
    if not resp and sections.get("intro"):
        intro_bul = bulleted(sections["intro"])
        # naive routing
        resp.extend(intro_bul[:6])
    return (req[:20], resp[:20], ben[:20])

def dedupe_lists(a: List[str], b: List[str]) -> Tuple[List[str], List[str]]:
    aset = set(x.lower() for x in a)
    b2 = [x for x in b if x.lower() not in aset]
    return (a, b2)

def years_required(text: str) -> Optional[int]:
    m = re.search(r"(\d{1,2})\+?\s*(?:years|yrs)\b", text, re.I)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            return None
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

def pick_allowlist(role_category: str) -> set:
    if role_category.startswith("Human Resources") or role_category.startswith("Trust & Safety"):
        return SKILLS_HR_TS_LIST
    return SKILLS_TECH_LIST

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
    # simple top keywords (could be TF-IDF later)
    words = re.findall(r"[a-zA-Z][a-zA-Z\-\+]{2,}", text.lower())
    freq: Dict[str,int] = {}
    stop = set("the and for with you your our this that from will are have has can into to of in on a an as by be is".split())
    for w in words:
        if w in stop: continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:25]
    return [w for w, _ in ranked]

# ---------- LLM client ----------
SESSION = requests.Session()

def llm_chat_json(prompt: str, text: str, keys_needed: List[str], max_tokens: int) -> Optional[Dict[str, Any]]:
    """Ask ONLY for missing keys; request JSON-only output."""
    sys_prompt = (
        "You extract structured fields from job descriptions. "
        "Return STRICT JSON with ONLY the requested keys. No prose, no code fences."
    )
    user = (
        f"Return JSON with EXACTLY these keys (and only these keys): {json.dumps(keys_needed)}.\n"
        "If a field is not explicitly present in the text, output null or [] accordingly.\n\n"
        f"TEXT:\n{text.strip()[:MAX_INPUT_CHARS]}"
    )
    payload = {
        "model": LLAMA_MODEL,
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt + "\n\n---\n" + user}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        # llama.cpp often supports OpenAI's response_format; it's safe to include
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

# ---------- QA ----------
def qa_flags_and_score(jd: Dict[str, Any]) -> Tuple[List[str], int]:
    flags = []
    # coverage
    if not jd.get("normalizedTitle"): flags.append("MISSING_TITLE")
    if not jd.get("seniorityLevel"): flags.append("MISSING_SENIORITY")
    if len(jd.get("responsibilitiesList") or []) < 3: flags.append("RESP_TOO_SHORT")
    # consistency
    yrs = jd.get("yearsExperienceRequired")
    sen = jd.get("seniorityLevel")
    if yrs is not None and sen in ("intern","junior") and yrs >= 2:
        flags.append("SENIORITY_INCONSISTENT")
    rm = jd.get("remoteMode")
    if rm not in (None, "remote", "hybrid", "onsite"):
        flags.append("REMOTE_ENUM_INVALID")
    # salary
    ct = jd.get("compensationText", "")
    cp = jd.get("compensationParsed")
    if ct and not (cp and (cp.get("min") or cp.get("max"))): flags.append("SALARY_PARSE_FAIL")
    # benefits
    if jd.get("benefitsRaw") and not jd.get("benefitsNormalized"):
        flags.append("BENEFITS_NOT_NORMALIZED")
    # skills baseline
    if len(jd.get("toolsTechnologies") or []) < 2 and jd.get("roleCategory","").startswith("Software"):
        flags.append("SKILLS_TOO_FEW")

    # score
    score = 100
    for fl in flags:
        score -= {
            "MISSING_TITLE":15, "MISSING_SENIORITY":10, "RESP_TOO_SHORT":10,
            "SENIORITY_INCONSISTENT":10, "REMOTE_ENUM_INVALID":10,
            "SALARY_PARSE_FAIL":10, "BENEFITS_NOT_NORMALIZED":5, "SKILLS_TOO_FEW":10
        }.get(fl, 5)
    return (flags, int(clamp(score, 0, 100)))

# ---------- Build 43-field skeleton ----------
def empty_43() -> Dict[str, Any]:
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

# ---------- Core enrich ----------
def enrich_one(job: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    raw = job.get("raw") or {}
    desc_html = job.get("description_html") or ""
    text = clean_html(desc_html)
    if not text: return None
    text_ck = checksum_text(text)

    jd = empty_43()

    # core title/seniority
    nt, sen = title_normalize(job.get("title") or "")
    jd["normalizedTitle"] = nt or None
    jd["titleAliases"] = [job.get("title")] if job.get("title") else []
    jd["seniorityLevel"] = sen or None

    # role category/specialty
    role_cat, role_spec = extract_role_category_and_specialty(job.get("title") or "", text)
    jd["roleCategory"] = role_cat
    jd["roleSpecialty"] = role_spec or None

    # employment type (from provider if present)
    emp = (job.get("employment_type") or "").lower() or (raw.get("job_type") or "").lower()
    jd["employmentType"] = emp if emp in ("full_time","part_time","contract","internship","temporary") else None

    # remote & regions
    remote_bool, remote_mode, regions = derive_remote(text, raw)
    jd["remoteMode"] = remote_mode
    jd["remoteEligibilityRegions"] = regions
    # location mentions
    loc_mentions = []
    for m in re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\b", text):
        # naive capture; UI uses remoteEligibilityRegions first anyway
        if m.lower() in ("we","our","you","the","and","or"): continue
        if len(m) >= 3: loc_mentions.append(m)
    jd["locationMentions"] = sorted(list(set(loc_mentions)))[:10]

    # sections
    sections = split_sections(text)
    req, resp, ben = extract_req_resp_benefits(sections)
    req, resp = dedupe_lists(req, resp)
    jd["responsibilitiesList"] = resp
    jd["benefitsRaw"] = ben
    jd["benefitsNormalized"] = normalize_benefits(ben)

    # years, seniority adjust
    yrs = years_required(text)
    jd["yearsExperienceRequired"] = yrs
    jd["seniorityLevel"] = adjust_seniority(jd["seniorityLevel"] or "", yrs) or jd["seniorityLevel"]

    # salary
    mn, mx, cur, per, est = parse_salary(text)
    if any([mn, mx, cur, per]):
        jd["compensationText"] = re.search(SAL_RE, text).group(0)
        jd["compensationParsed"] = {"min": mn, "max": mx, "currencySymbol": cur, "period": per}

    # languages
    jd["languageRequirements"] = parse_languages(text)

    # tools & methods (allowlisted)
    allow = pick_allowlist(role_cat)
    seeded = seed_skills(text, (raw.get("tags") or job.get("tags") or []), allow)
    jd["toolsTechnologies"] = seeded

    # methods/frameworks basic cues
    meth = []
    if re.search(r"\b(agile|scrum|kanban)\b", text, re.I): meth.append("agile")
    if re.search(r"\b(tdd|testing)\b", text, re.I): meth.append("testing")
    if re.search(r"\b(mlops)\b", text, re.I): meth.append("mlops")
    jd["methodsFrameworks"] = sorted(list(set(meth)))

    # domain/product area
    jd["industryDomain"] = list(sorted(set([x for x in [role_cat.split(" / ")[-1], "recruitment services" if "recruit" in text.lower() else "", "renewable energy" if "energy" in text.lower() else ""] if x])))
    jd["productArea"] = []

    # compliance & sensitivity (lightweight)
    if re.search(r"\b(gdpr|hipaa|soc2|iso)\b", text, re.I):
        jd.setdefault("complianceRegulations", []).extend(sorted(set(re.findall(r"\b(gdpr|hipaa|soc2|iso)\b", text, re.I))))
    if re.search(r"\b(content|moderation|trust)\b", text, re.I):
        jd["dataSensitivity"] = ["content moderation"] if "moderation" in text.lower() else []

    # perks/culture/process
    if "retreat" in text.lower(): jd.setdefault("perksExtras", []).append("retreat")
    jd["cultureSignals"] = []
    if "founder-led" in text.lower(): jd["cultureSignals"].append("founder-led")
    if "remote-first" in text.lower(): jd["cultureSignals"].append("remote-first")
    jd["cultureSignals"] = sorted(list(set(jd["cultureSignals"])))
    jd["deiStatementPresence"] = True if re.search(r"equal opportunity|diversity|all backgrounds", text, re.I) else False
    jd["applicationProcess"] = ["apply via platform"] if re.search(r"\bapply\b", text, re.I) else []
    jd["workEnvironment"] = ["remote"] if remote_mode in ("remote","hybrid") else []
    jd["keywordsTaxonomy"] = keywords_taxonomy(text)

    # --- LLM micro-call for missing bits (summary, outcomes, extras) ---
    need_keys = []
    if len(jd["responsibilitiesList"]) < 3: need_keys.append("responsibilitiesList")
    if len(req) < 3: need_keys.append("requirements")
    if not jd.get("workHoursWindow") and re.search(r"\b(overlap|CET|GMT|PST|EST)\b", text): need_keys.append("workHoursWindow")
    if not jd.get("outcomesKpis"): need_keys.append("outcomesKpis")
    if "summary_tldr" not in need_keys: need_keys.append("summary_tldr")

    llm_added = {}
    if need_keys:
        # cache key
        cache_key = f"{LLAMA_MODEL}:{PROMPT_VERSION}:{text_ck}:{','.join(sorted(need_keys))}"
        cached = cache_get(cache_key)
        if cached:
            llm_added = cached
        else:
            prompt = PROMPT_MISSING_PATH.read_text(encoding="utf-8")
            llm_added = llm_chat_json(prompt, text, need_keys, MAX_OUT_TOKENS) or {}
            if llm_added: cache_put(cache_key, llm_added)

    # merge LLM fields cautiously
    if isinstance(llm_added.get("responsibilitiesList"), list) and len(jd["responsibilitiesList"]) < 3:
        jd["responsibilitiesList"] = dedupe_lists(jd["responsibilitiesList"], llm_added["responsibilitiesList"])[0]
    if isinstance(llm_added.get("requirements"), list) and len(req) < 3:
        req = llm_added["requirements"][:20]
    # work hours
    if isinstance(llm_added.get("workHoursWindow"), str) and not jd.get("workHoursWindow"):
        jd["workHoursWindow"] = llm_added["workHoursWindow"][:80]
    # outcomes
    if isinstance(llm_added.get("outcomesKpis"), list) and not jd.get("outcomesKpis"):
        jd["outcomesKpis"] = llm_added["outcomesKpis"][:10]
    # summary
    summary = llm_added.get("summary_tldr") or ""
    summary = summary.strip()
    # summary we store at top-level convenience later
    # add requirements finally
    jd["requirementsList"] = req  # keep for merging to legacy fields later

    # finalize benefitsNormalized if missing
    if jd["benefitsRaw"] and not jd["benefitsNormalized"]:
        jd["benefitsNormalized"] = normalize_benefits(jd["benefitsRaw"])

    # QA
    flags, score = qa_flags_and_score(jd)
    meta = {
        "modelVersion": MODEL_VERSION,
        "promptVersion": PROMPT_VERSION,
        "textChecksum": text_ck,
        "enrichedAt": now_iso(),
    }

    # Legacy convenience fields
    legacy = {
        "normalized_title": jd["normalizedTitle"],
        "seniority": jd["seniorityLevel"],
        "summary_tldr": summary[:240] if summary else "",
        "remote_mode": jd["remoteMode"],
        "quality_score": score
    }

    result = {
        "jd_v3": jd,
        "jd_v3_meta": meta,
        "jd_v3_qa": {"score": score, "flags": flags},
        "legacy": legacy
    }
    return result

# ---------- Selection & writing ----------
def build_selector() -> Dict[str, Any]:
    rule = (CFG.get("selector", {}) or {}).get("rule", "enriched_at_missing_or_stale")
    if rule == "enriched_at_missing_or_stale":
        return {
            "$or": [
                {"jd_v3_meta": {"$exists": False}},
                {"$expr": {"$gt": ["$last_seen_at", "$jd_v3_meta.enrichedAt"]}}
            ]
        }
    return {}

def projection():
    return {
        "_id": 1, "title": 1, "company": 1, "source": 1, "raw": 1,
        "employment_type": 1, "description_html": 1, "tags": 1,
        "last_seen_at": 1, "jd_v3_meta": 1
    }

def process_docs(docs: List[Dict[str, Any]]) -> Dict[str, int]:
    ops = []
    updated = 0
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        fut = {ex.submit(enrich_one, d): d for d in docs}
        for f in as_completed(fut):
            doc = fut[f]
            try:
                out = f.result()
            except Exception as e:
                print(f"[ENRICH ERROR] {e}")
                out = None
            if not out: continue
            set_fields = {
                "jd_v3": out["jd_v3"],
                "jd_v3_meta": out["jd_v3_meta"],
                "jd_v3_qa": out["jd_v3_qa"],
                # legacy mirrors for compatibility with v2 UI
                "normalized_title": out["legacy"]["normalized_title"],
                "seniority": out["legacy"]["seniority"],
                "summary_tldr": out["legacy"]["summary_tldr"],
                "remote_mode": out["legacy"]["remote_mode"],
                "quality_score": out["legacy"]["quality_score"],
            }
            ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": set_fields}))
            updated += 1
    if ops:
        jobs_col.bulk_write(ops, ordered=False)
    return {"updated": updated}

def count_to_process() -> int:
    return jobs_col.count_documents(build_selector())

def render_progress(done: int, total: int, width: int = 28) -> str:
    if total <= 0: return f"{done} / ?"
    frac = min(1.0, done/total)
    filled = int(round(frac * width))
    bar = "█" * filled + " " * (width - filled)
    return f"{bar}  {done:,} / {total:,} ({frac*100:4.1f}%)"

# ---------- CLI ----------
import argparse
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", dest="job_id", help="Process only this job _id")
    ap.add_argument("--limit", type=int, default=None, help="Override batch size")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    return ap.parse_args()

def main():
    args = parse_args()
    q = build_selector()
    if args.job_id:
        q = {"_id": args.job_id}
    total = jobs_col.count_documents(q)
    if args.limit: total = min(total, args.limit)

    processed = 0
    while True:
        lim = args.limit or min(BATCH_SIZE, total - processed) if total else (args.limit or BATCH_SIZE)
        cur = list(jobs_col.find(q, projection=projection(), limit=lim))
        if not cur: break
        res = process_docs(cur)
        processed += res["updated"]
        if not args.no_progress:
            sys.stdout.write("\r" + render_progress(processed, total))
            sys.stdout.flush()
        if args.job_id: break
        if args.limit and processed >= args.limit: break
        if MAX_JOBS_TOTAL and processed >= MAX_JOBS_TOTAL: break
    if not args.no_progress: sys.stdout.write("\n")
    print(json.dumps({"ok": True, "processed": processed, "total": total, "db": DB_NAME}, indent=2))

if __name__ == "__main__":
    main()
