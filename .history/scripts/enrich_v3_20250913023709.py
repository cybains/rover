#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
enrich_v3.py  —  v3.1 (fix pack)
- Windowed 100-job batches with 70/15/15 split
- Resume-safe; auto-skip on error/timeout; optional retry
- Prints original HTML (truncated), cleaned text, and enriched fields after each success
- Rule-first extraction, LLM micro-call only when needed (cached)
- FIXES:
  • Sectioning: recognizes “What You’ll Do”, “Key Responsibilities”, “The impact you’ll have here”, “What You Bring”, etc.
  • Salary: ignores “2–5 years” ranges; prefers currency/unit; parses hourly like “$4–$7 USD/hr”
  • Taxonomy: adds Marketing/Content & Infrastructure/DevOps/Platform
  • Remote/location: parses “can be based remotely in US or Europe”, pulls “Location:” label, filters location noise
  • Language: no more Italian from “it”; only full names & safe codes
  • Skills: adds BI/marketing & infra stacks (HubSpot, SEO, Terraform, Prometheus, etc.)
  • Compliance: HIPAA, HITRUST, SOC 2, ISO 27001
  • Work hours/on-call: captures PST overlaps & on-call rotations
  • QA flags for the above issues
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

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# --------------------------- Env / DB ---------------------------

MONGO_URI = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
if not MONGO_URI:
    print("ERROR: Set MONGO_URI (or MONGODB_URI) in your .env")
    sys.exit(1)

DB_NAME = os.environ.get("JOBS_DB_NAME", "refjobs")
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
jobs_col = db["jobs"]
cache_col = db.get_collection("jobs_llm_cache")

# --------------------------- Optional deps ---------------------------

try:
    from bs4 import BeautifulSoup, NavigableString
except Exception:
    BeautifulSoup = None  # script still works with text-only fallback

# --------------------------- Load Config ---------------------------

DEFAULT_CFG = {
    "version": 3,
    "model_version": "v3.1",
    "prompt_version": "v3.1",
    "selector": {"rule": "windowed_100"},
    "batch": {"size": 100, "max_concurrency": 2, "max_jobs_total": 0},
    "timeouts": {"llm": 300},
    "llm": {
        "base_url": "http://127.0.0.1:8080",
        "model": "Phi-3.5-mini-instruct-Q5_K_S.gguf",
        "temperature": 0.1,
        "max_input_chars": 3500,
        "max_output_tokens": 256
    }
}

CFG = DEFAULT_CFG
if CFG_PATH.exists():
    try:
        user_cfg = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8")) or {}
        CFG.update(user_cfg)
    except Exception as e:
        print(f"[CFG WARN] Could not read {CFG_PATH}: {e}. Using defaults.")

LLM = CFG.get("llm", {})
LLAMA_BASE_URL = LLM.get("base_url", DEFAULT_CFG["llm"]["base_url"])
LLAMA_MODEL = LLM.get("model", DEFAULT_CFG["llm"]["model"])
TEMPERATURE = float(LLM.get("temperature", 0.1))
MAX_INPUT_CHARS = int(LLM.get("max_input_chars", 3500))
MAX_OUT_TOKENS = int(LLM.get("max_output_tokens", 256))
PROMPT_VERSION = CFG.get("prompt_version", "v3.1")
MODEL_VERSION = CFG.get("model_version", "v3.1")
LLM_TIMEOUT = int(CFG.get("timeouts", {}).get("llm", 300))

BATCH_SIZE = int(CFG.get("batch", {}).get("size", 100))
MAX_CONCURRENCY = int(CFG.get("batch", {}).get("max_concurrency", 2))
MAX_JOBS_TOTAL = int(CFG.get("batch", {}).get("max_jobs_total", 0))

# --------------------------- CLI ---------------------------

import argparse
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", dest="job_id", help="Process only this job _id")
    ap.add_argument("--retry-failures", action="store_true", help="Retry prior failures/timeouts before assigning new window")
    ap.add_argument("--window-size", type=int, default=BATCH_SIZE, help="Window size (default 100)")
    ap.add_argument("--ask-on-error", action="store_true", help="Ask to skip/retry on error (auto-skip after 5 min)")
    ap.add_argument("--print", dest="print_all", action="store_true", help="Print full jd_v3 after each successful enrichment")
    ap.add_argument("--print-fields", dest="print_fields", help="Comma-separated subset of jd_v3 keys to print")
    ap.add_argument("--print-html", action="store_true", help="Print CLEANED TEXT under each job")
    ap.add_argument("--print-raw-html", action="store_true", help="Print ORIGINAL HTML (truncated) under each job")
    ap.add_argument("--no-progress", action="store_true", help="Disable progress bar")
    return ap.parse_args()

ARGS = parse_args()
PRINT_ALL = bool(ARGS.print_all)
PRINT_FIELDS = [k.strip() for k in (ARGS.print_fields or "").split(",") if k.strip()] or None
PRINT_HTML = bool(ARGS.print_html)
PRINT_RAW_HTML = bool(ARGS.print_raw_html)
if PRINT_ALL or PRINT_FIELDS or PRINT_HTML or PRINT_RAW_HTML:
    ARGS.no_progress = True

SESSION = requests.Session()

def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"

def clean_html_text(html_str: Optional[str]) -> str:
    if not html_str: return ""
    if BeautifulSoup is not None:
        try:
            soup = BeautifulSoup(html_str, "lxml")
            for s in soup(["script","style"]): s.extract()
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

# --------------------------- Header detection & sections ---------------------------

RESP_HEADERS = [
    "responsibilities", "what you will do", "what you’ll do", "key responsibilities",
    "the impact you’ll have here", "impact you will have", "impact you'll have"
]
REQ_HEADERS = [
    "requirements", "qualifications", "what we’re looking for", "what we're looking for",
    "about you", "what you bring"
]
BEN_HEADERS = [
    "benefits", "perks", "why join", "why [", "life at "
]

def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()

def is_header_text(txt: str) -> Optional[str]:
    t = (txt or "").strip().lower().rstrip(":")
    for h in RESP_HEADERS:
        if t.startswith(h): return "resp"
    for h in REQ_HEADERS:
        if t.startswith(h): return "req"
    for h in BEN_HEADERS:
        if t.startswith(h) or t.startswith("why join"): return "ben"
    return None

def text_bulleted(segment: str) -> List[str]:
    if not segment: return []
    out = []
    for raw in segment.split("\n"):
        s = raw.strip()
        if not s: continue
        s = s.strip("•-–*·:\t ").strip()
        if len(s) >= 3:
            out.append(s)
    # de-dup
    uniq, seen = [], set()
    for x in out:
        xl = x.lower()
        if xl not in seen:
            seen.add(xl); uniq.append(x)
    return uniq

def parse_sections_from_html(html_str: str) -> Dict[str, List[str]]:
    """
    Prefer bullets immediately under known headers. Falls back to text splitter.
    """
    if not html_str or BeautifulSoup is None:
        return {}

    soup = BeautifulSoup(html_str, "lxml")

    def header_nodes():
        # headers often appear as <p><strong>Header</strong></p>, or <h2>, <h3>
        for tag in soup.find_all(["h1","h2","h3","h4","p","strong","b"]):
            txt = norm(tag.get_text(" "))
            role = is_header_text(txt)
            if role:
                yield (role, tag)

    buckets = {"resp": [], "req": [], "ben": []}
    for role, node in header_nodes():
        # collect bullets from the next sibling lists or until next header
        items = []
        # step 1: immediate <ul>/<ol> after header
        nxt = node
        for _ in range(8):  # look ahead a few siblings
            nxt = nxt.find_next_sibling()
            if not nxt: break
            # stop if another header encountered
            if is_header_text(norm(nxt.get_text(" "))) is not None:
                break
            if nxt.name in ("ul","ol"):
                for li in nxt.find_all("li"):
                    s = norm(li.get_text(" "))
                    if s: items.append(s)
            # also capture short paragraphs that look like bullets
            if nxt.name in ("p","div") and not nxt.find(["ul","ol"]):
                s = norm(nxt.get_text(" "))
                # avoid capturing labels like "Location:"
                if s and not s.lower().startswith(("location:", "about the company", "about dbeaver", "about us", "about")):
                    # treat lines with many colons as headings; skip those
                    if s.count(":") <= 2 and len(s) <= 240:
                        items.append(s)
        if items:
            buckets[role].extend(items)

    # de-dup & trim noisy colon-only headers
    for k in buckets:
        cleaned = []
        for s in buckets[k]:
            ss = s.strip().strip(":").strip()
            if not ss: continue
            # drop common headings that slipped in
            if ss.lower() in ("what you’ll do","what you'll do","requirements","qualifications","what you bring",
                              "key responsibilities","the impact you’ll have here","why join"):
                continue
            cleaned.append(ss)
        # preserve order but remove dup
        seen = set(); ded = []
        for s in cleaned:
            lo = s.lower()
            if lo not in seen:
                seen.add(lo); ded.append(s)
        buckets[k] = ded[:30]

    return buckets

def split_sections_from_text(text: str) -> Dict[str, List[str]]:
    """
    Fallback: rough split by headings in plain text.
    """
    segs = {"resp": [], "req": [], "ben": []}
    blocks = re.split(r"\n{2,}", text or "")
    current = None
    for blk in blocks:
        head = is_header_text(blk.split("\n",1)[0] if blk else "")
        if head:
            current = head
            # remove the header line
            lines = blk.split("\n")
            payload = "\n".join(lines[1:]) if len(lines) > 1 else ""
            segs[current].extend(text_bulleted(payload))
        else:
            if current:
                segs[current].extend(text_bulleted(blk))
    # prune obvious junk
    for k in segs:
        segs[k] = [x for x in segs[k] if not x.lower().startswith(("location:", "why join", "about the company"))]
        # remove single-word pseudo bullets
        segs[k] = [x for x in segs[k] if len(x.split()) >= 3]
        segs[k] = segs[k][:30]
    return segs

# --------------------------- Title/role taxonomy ---------------------------

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

def extract_role_category_and_specialty(title: str, text: str) -> Tuple[str, str]:
    t = f"{title} {text}".lower()

    # Marketing / Content
    if re.search(r"\b(copywriter|content (writer|marketer)|seo|social media|email marketing)\b", t):
        return ("Marketing / Content", "B2B SaaS")

    # Infrastructure / DevOps / Platform
    if re.search(r"\b(devops|sre|site reliability|platform engineer|infrastructure engineer|kubernetes|terraform|helm|prometheus|grafana)\b", t):
        return ("Infrastructure / DevOps / Platform", "Platform/DevOps")

    # Trust & Safety
    if re.search(r"\b(trust\s*&?\s*safety|moderation|content policy)\b", t):
        return ("Trust & Safety", "Content Moderation")

    # HR / Recruiting
    if re.search(r"\b(recruiter|talent acquisition|sourcing)\b", t):
        return ("Human Resources", "Talent Acquisition")

    # Software / Data (data analysts, DS, ML)
    if re.search(r"\b(data analyst|data scientist|machine learning|ml|analytics)\b", t):
        return ("Software / Data", "Data Science/Analytics")

    # Software / Dev (general)
    if re.search(r"\b(software|backend|frontend|full[- ]stack|api|node\.?js|typescript|javascript|retool|postgres)\b", t):
        return ("Software / Dev", "Backend/Platform")

    # Logistics / Transport
    if re.search(r"\b(truck(ing)? driver|cdl|hgv|courier|delivery driver|last[- ]mile|otr|tanker|flatbed|box truck)\b", t):
        return ("Logistics / Transport", "Truck Driver")

    return ("Other", "")

# --------------------------- Remote & location parsing ---------------------------

REMOTE_PATTERNS = [
    re.compile(r"\bfully remote\b", re.I),
    re.compile(r"\bremote[- ]first\b", re.I),
    re.compile(r"\bwork from home\b", re.I),
    re.compile(r"\banywhere\b", re.I),
    re.compile(r"\bcan be based remotely\b", re.I),
    re.compile(r"\b(remote)\b", re.I),
]

HYBRID_PATTERNS = [
    re.compile(r"\bhybrid\b", re.I),
    re.compile(r"\b\d+\s+days?\s+onsite\b", re.I),
    re.compile(r"\b\d{1,3}%\s*onsite\b", re.I),
]

ONSITE_PATTERNS = [
    re.compile(r"\bonsite\b", re.I),
    re.compile(r"\bon[- ]site\b", re.I),
]

COUNTRIES = set(map(str.lower, """
United States, USA, U.S., U.S.A., America, Europe, Serbia, Hungary, Philippines, Portugal, Italy, Germany, France, Spain,
United Kingdom, UK, Ireland, Netherlands, Belgium, Austria, Switzerland, Canada, Mexico, Brazil, Argentina, Chile, Poland,
Czech Republic, Slovakia, Slovenia, Croatia, Romania, Bulgaria, Greece, Turkey, Sweden, Norway, Denmark, Finland, Estonia,
Latvia, Lithuania, India, Pakistan, Bangladesh, Sri Lanka, Nepal, China, Japan, South Korea, Singapore, Malaysia, Thailand,
Vietnam, Indonesia, Australia, New Zealand, South Africa, Nigeria, Kenya, Egypt, UAE, Saudi Arabia, Israel
""".replace("\n","").split(",")))

CITIES_HINT = set(map(str.lower, "Belgrade,Budapest,Boston,London,Paris,Berlin,Amsterdam,Vienna,Zurich,Prague".split(",")))

def parse_location_label(html_str: str, text: str) -> List[str]:
    out = []
    # HTML label "Location:" → capture right-hand side
    if html_str and BeautifulSoup is not None:
        soup = BeautifulSoup(html_str, "lxml")
        for strong in soup.find_all(["strong","b"]):
            st = norm(strong.get_text(" "))
            if st.lower().startswith("location"):
                # get next text
                tail = strong.parent.get_text(" ") if strong.parent else st
                after = tail.split(":",1)[1] if ":" in tail else ""
                after = norm(after)
                if after:
                    out.extend([x.strip() for x in re.split(r"[;/,]| or ", after) if x.strip()])
    # fallback: text line starting with Location:
    for m in re.finditer(r"^location\s*:\s*(.+)$", text, re.I|re.M):
        line = norm(m.group(1))
        parts = [x.strip() for x in re.split(r"[;/,]| or ", line)]
        out.extend([p for p in parts if p])

    # filter: keep countries/known cities/regions words
    cleaned = []
    for token in out:
        tl = token.lower()
        if tl in COUNTRIES or tl in CITIES_HINT:
            cleaned.append(token)
        elif "europe" in tl or "us" == tl or "usa" == tl or "united states" in tl:
            cleaned.append(token)
    # unique
    seen=set(); final=[]
    for v in cleaned:
        k=v.lower()
        if k not in seen:
            seen.add(k); final.append(v)
    return final[:8]

def derive_remote_and_regions(text: str, html_str: str, provider: Dict[str, Any]) -> Tuple[bool, str, List[str]]:
    t = text.lower()
    remote, mode = False, "onsite"
    if any(p.search(t) for p in REMOTE_PATTERNS):
        remote, mode = True, "remote"
    if any(p.search(t) for p in HYBRID_PATTERNS):
        remote, mode = True, "hybrid"
    if any(p.search(t) for p in ONSITE_PATTERNS):
        remote, mode = False, "onsite"

    regions = []
    # “can be based remotely in the United States or Europe”
    m = re.search(r"based\s+remotely\s+in\s+(.+?)(?:\.|,|;|$)", t)
    if m:
        blob = m.group(1)
        # split on or/and/commas
        parts = re.split(r"\s*(?:,|or|and)\s*", blob)
        for p in parts:
            pp = p.strip(" .")
            if not pp: continue
            if pp in ("the united states","united states","us","usa","u.s.","u.s.a."):
                regions.append("United States")
            elif "europe" in pp:
                regions.append("Europe")
            elif pp.title() in COUNTRIES:
                regions.append(pp.title())

    # provider hint
    loc = (provider or {}).get("candidate_required_location")
    if loc:
        regions.append(loc)

    # Location label (offices) may mention office cities; we’ll store as location mentions, not eligibility
    # but if remote and we parsed “United States/Europe” from label, include those
    locs_from_label = parse_location_label(html_str, text)
    for v in locs_from_label:
        vl = v.lower()
        if "united states" in vl or vl in ("usa","us") or "europe" in vl:
            regions.append("United States" if "united states" in vl or vl in ("usa","us") else "Europe")

    # unique
    uniq=[]; seen=set()
    for r in regions:
        if not r: continue
        k=r.lower()
        if k not in seen:
            seen.add(k); uniq.append(r)
    return ((mode in ("remote","hybrid")), mode, uniq[:6])

# --------------------------- Salary parsing (guardrails) ---------------------------

CUR_MAP = {"$":"USD","€":"EUR","£":"GBP"}
UNIT_WORDS = r"(?:per\s+hour|per\s+year|per\s+month|/hr|/hour|/yr|/year|/mo|/month|annual|hourly|monthly)"
CURR_WORDS = r"(?:usd|eur|gbp|dollars|euros|pounds)"

RANGE_RE = re.compile(
    rf"(?P<cur>[$€£])?\s*(?P<min>\d[\d,\.kK]*)\s*[-–—]\s*(?P<max>\d[\d,\.kK]*)\s*(?P<unit>{UNIT_WORDS})?\s*(?P<iso>{CURR_WORDS})?",
    re.I
)

def _num(s: str) -> Optional[float]:
    try:
        s = s.lower().replace(",", "").strip()
        if s.endswith("k"): return float(s[:-1]) * 1000.0
        return float(s)
    except Exception:
        return None

def _near_years(txt: str, start: int, end: int) -> bool:
    window = txt[max(0,start-20):min(len(txt), end+20)].lower()
    return bool(re.search(r"\b(year|years|yr|yrs)\b", window))

def parse_salary(text: str) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str], Optional[str], bool]:
    """
    Returns (min, max, currency, period, raw_text, ctx_mismatch_flag)
    Picks best candidate: currency > unit > neither. Discards ranges near "years/yrs".
    """
    if not text: return (None,None,None,None,None,False)
    best = None
    mismatch = False
    for m in RANGE_RE.finditer(text):
        s, e = m.span()
        if _near_years(text, s, e):
            mismatch = True
            continue
        mn = _num(m.group("min")); mx = _num(m.group("max"))
        if mn is None or mx is None: continue
        cur_sym = (m.group("cur") or "").strip()
        cur = CUR_MAP.get(cur_sym) if cur_sym else None
        iso = (m.group("iso") or "").lower().strip()
        if iso in ("usd","eur","gbp"): cur = iso.upper()
        unit = (m.group("unit") or "").lower()

        # Score: currency present 2, unit present 1
        score = (2 if cur else 0) + (1 if unit else 0)
        # prefer hourly if both present & unit contains hour
        if best is None or score > best[0] or (score == best[0] and (("hour" in unit) and ("hour" not in (best[3] or "")))):
            best = (score, (mn, mx), cur, unit, m.group(0))
    if not best: return (None,None,None,None,None,mismatch)
    mn, mx = best[1]
    cur = best[2]
    unit = best[3] or ""
    period = None
    if "year" in unit or "annual" in unit or "/yr" in unit: period = "year"
    elif "month" in unit or "/mo" in unit: period = "month"
    elif "hour" in unit or "/hr" in unit: period = "hour"
    return (mn, mx, cur, period, best[4], mismatch)

# --------------------------- Language parsing (safe) ---------------------------

LANG_MAP = {
    "english":"en","eng":"en",
    "german":"de","deutsch":"de",
    "french":"fr","français":"fr","francais":"fr",
    "italian":"it","italiano":"it",
    "dutch":"nl","nederlands":"nl",
    "spanish":"es","español":"es","espanol":"es",
    "portuguese":"pt","português":"pt","portugues":"pt",
    "polish":"pl","czech":"cs","slovak":"sk","hungarian":"hu","serbian":"sr"
}
CEFR_LVLS = ["C2","C1","B2","B1","A2","A1"]

def parse_languages(text: str) -> List[Dict[str, Any]]:
    out = []
    t = text.lower()
    for lang_name, code in LANG_MAP.items():
        # require full language word (no bare "it")
        if re.search(rf"\b{re.escape(lang_name)}\b", t, re.I):
            lvl = None
            for L in CEFR_LVLS:
                if re.search(rf"\b{L}\b", t, re.I):
                    lvl = L; break
            out.append({"lang": code, "level": lvl, "source": "desc", "confidence": 0.7 if lvl else 0.5})
    # unique by lang
    uniq=[]; seen=set()
    for d in out:
        if d["lang"] not in seen:
            seen.add(d["lang"]); uniq.append(d)
    return uniq[:6]

# --------------------------- Skills allowlists ---------------------------

def load_yaml_list(p: Path) -> List[str]:
    if not p.exists(): return []
    y = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    xs = y.get("skills", []) if isinstance(y, dict) else (y or [])
    return [str(s).strip().lower() for s in xs if str(s).strip()]

# Fallback defaults (used if no external ontology files provided)
SKILLS_TECH_DEFAULT = set("""
python,sql,postgres,mysql,git,linux,docker,kubernetes,aws,azure,gcp,flask,fastapi,django,pytorch,tensorflow,airflow,kafka,bash,mlops,terraform,helm,ansible,prometheus,grafana,elk,elasticsearch,logstash,kibana,iam,oauth2,jwt,api gateway,prefect,postgresql
""".replace("\n","").split(","))

SKILLS_HR_TS_DEFAULT = set("""
recruiting,sourcing,ats,interviewing,stakeholder management,content moderation,policy,triage,case management
""".replace("\n","").split(","))

SKILLS_MARKETING_DEFAULT = set("""
hubspot,seo,sem,cms,wordpress,webflow,google analytics,copywriting,content marketing,social media,linkedin,x,figma,jira,g suite,slack,email marketing,crm
""".replace("\n","").split(","))

def pick_allowlist(role_category: str) -> set:
    rc = role_category.lower()
    if rc.startswith("human resources") or rc.startswith("trust & safety"):
        return SKILLS_HR_TS_DEFAULT
    if rc.startswith("marketing"):
        return SKILLS_MARKETING_DEFAULT
    if rc.startswith("infrastructure"):
        return SKILLS_TECH_DEFAULT
    if rc.startswith("software"):
        return SKILLS_TECH_DEFAULT
    return SKILLS_TECH_DEFAULT

def seed_skills(text: str, provider_tags: List[str], allow: set) -> List[str]:
    out = set()
    for t in provider_tags or []:
        tt = t.strip().lower()
        if tt in allow: out.add(tt)
    for sk in allow:
        if re.search(rf"\b{re.escape(sk)}\b", text, re.I):
            out.add(sk)
    return sorted(out)

# --------------------------- Compliance, work hours, on-call ---------------------------

def parse_compliance(text: str) -> List[str]:
    out = set()
    t = text.lower()
    if "hipaa" in t: out.add("HIPAA")
    if "hitrust" in t: out.add("HITRUST")
    if "soc 2" in t or "soc2" in t: out.add("SOC 2")
    if re.search(r"\biso\s*27001\b", t): out.add("ISO 27001")
    return sorted(out)

def parse_work_hours(text: str) -> Optional[str]:
    # Capture overlaps like "6 hours of overlap with PST (8 am to 2 pm PST)"
    m = re.search(r"(\d+)\s+hours?\s+of\s+overlap\s+with\s+([A-Z]{2,4})(?:.*?\b(\d{1,2}\s*[:.]?\s*\d*\s*(?:am|pm))\s*to\s*(\d{1,2}\s*[:.]?\s*\d*\s*(?:am|pm))\s*\2)?", text, re.I)
    if m:
        hrs = m.group(1); tz = m.group(2)
        if m.group(3) and m.group(4):
            return f"{hrs}h overlap with {tz} ({m.group(3)}–{m.group(4)} {tz})"
        return f"{hrs}h overlap with {tz}"
    return None

def parse_on_call(text: str) -> Optional[str]:
    if re.search(r"\bon[- ]?call\b", text, re.I):
        return "On-call rotation"
    return None

# --------------------------- LLM (optional micro-call) ---------------------------

def load_missing_prompt() -> str:
    if PROMPT_MISSING_PATH.exists():
        try:
            return PROMPT_MISSING_PATH.read_text(encoding="utf-8")
        except Exception:
            pass
    return ("Extract ONLY the requested keys from the job text.\n"
            "- Arrays concise (<= 8). If missing, return [] or null.\n"
            "- No invented salaries/dates/locations. JSON only.\n"
            "Keys: responsibilitiesList, requirements, workHoursWindow, outcomesKpis, summary_tldr\n")

def llm_chat_json(text: str, keys_needed: List[str]) -> Optional[Dict[str, Any]]:
    sys_prompt = ("You extract structured fields from job descriptions. "
                  "Return STRICT JSON with ONLY the requested keys.")
    user = (
        f"Return JSON with EXACTLY these keys: {json.dumps(keys_needed)}.\n"
        "If a field is not explicitly present, output null or [].\n\n"
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

def qa_flags_and_score(jd: Dict[str, Any], aux: Dict[str, Any]) -> Tuple[List[str], int]:
    flags = []
    # Responsibilities/requirements contamination
    bad_heads = ("location:", "what you’ll do", "what you'll do", "why join", "about the company", "about dbeaver")
    if any((x.lower().startswith(bh) for x in (jd.get("responsibilitiesList") or []) for bh in bad_heads)):
        flags.append("RESP_CONTAINS_HEADERS")
    if any(("remote-first" in x.lower() or "competitive compensation" in x.lower())
           for x in (jd.get("requirementsList") or [])):
        flags.append("REQS_CONTAINS_BENEFITS")
    # Salary mismatch
    if aux.get("salary_ctx_mismatch"):
        flags.append("SALARY_CONTEXT_MISMATCH")
    # Language FP (Italian from 'it')
    if any((d.get("lang")=="it" and "ital" not in aux.get("cleaned_text","").lower()) for d in jd.get("languageRequirements") or []):
        flags.append("LANGUAGE_FP_IT")
    # Location noise: too many non-places → basic heuristic: if >8 mentions or none from COUNTRIES
    locs = jd.get("locationMentions") or []
    if len(locs) > 8:
        flags.append("LOCATION_NOISE")
    if not any((l.lower() in COUNTRIES or l.lower() in CITIES_HINT or "united states" in l.lower() or "europe" in l.lower()) for l in locs) and locs:
        flags.append("LOCATION_NOISE")
    # Missing work hours where overlap present in text but not captured
    if "overlap" in aux.get("cleaned_text","").lower() and not jd.get("workHoursWindow"):
        flags.append("MISSING_WORK_HOURS")
    # Simple score
    score = 100
    penalties = {
        "RESP_CONTAINS_HEADERS":10, "REQS_CONTAINS_BENEFITS":10, "SALARY_CONTEXT_MISMATCH":10,
        "LANGUAGE_FP_IT":5, "LOCATION_NOISE":5, "MISSING_WORK_HOURS":5
    }
    for fl in flags: score -= penalties.get(fl, 5)
    # Add bonuses
    if len(jd.get("toolsTechnologies") or []) >= 5: score += 5
    if jd.get("remoteMode") in ("remote","hybrid"): score += 3
    score = max(0, min(100, score))
    return (flags, score)

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

# --------------------------- Utilities ---------------------------

def keywords_taxonomy(text: str) -> List[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z\-\+]{2,}", text.lower())
    freq: Dict[str,int] = {}
    stop = set("the and for with you your our this that from will are have has can into to of in on a an as by be is we they".split())
    for w in words:
        if w in stop: continue
        freq[w] = freq.get(w, 0) + 1
    ranked = sorted(freq.items(), key=lambda x: (-x[1], x[0]))[:25]
    return [w for w, _ in ranked]

def dedupe(seq: List[str]) -> List[str]:
    out=[]; seen=set()
    for s in seq:
        k=s.lower()
        if k not in seen:
            seen.add(k); out.append(s)
    return out

def parse_years_range(text: str) -> Tuple[Optional[int], Optional[int]]:
    # match 2-5 years, 2 – 5 yrs, 3+ years
    m = re.search(r"(\d{1,2})\s*[-–—]\s*(\d{1,2})\s*\+?\s*(?:years?|yrs?)", text, re.I)
    if m:
        lo = int(m.group(1)); hi = int(m.group(2))
        if lo <= hi: return lo, hi
    m = re.search(r"(\d{1,2})\s*\+?\s*(?:years?|yrs?)", text, re.I)
    if m:
        v=int(m.group(1)); return v, None
    return None, None

def adjust_seniority(initial: str, yrs_min: Optional[int], yrs_max: Optional[int], title: str) -> str:
    if initial: return initial
    # heuristic: use min years primarily
    y = yrs_min if yrs_min is not None else (yrs_max if yrs_max is not None else None)
    if y is None: return ""
    if y >= 5: return "senior" if re.search(r"\bsenior|lead|staff|principal\b", title.lower()) else "mid"
    if y >= 2: return "mid"
    return "junior"

# --------------------------- Enrichment core ---------------------------

def enrich_one(job: Dict[str, Any]) -> Dict[str, Any]:
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
    # role category/specialty
    role_cat, role_spec = extract_role_category_and_specialty(job.get("title") or "", cleaned)
    jd["roleCategory"] = role_cat
    jd["roleSpecialty"] = role_spec or None

    # employment type
    emp = (job.get("employment_type") or "").lower() or (raw.get("job_type") or "").lower()
    jd["employmentType"] = emp if emp in ("full_time","part_time","contract","internship","temporary") else None

    # remote + regions
    _, remote_mode, regions = derive_remote_and_regions(cleaned, desc_html, raw)
    jd["remoteMode"] = remote_mode
    jd["remoteEligibilityRegions"] = regions

    # location mentions: prefer explicit "Location:" label; also keep office cities if present
    locs = parse_location_label(desc_html, cleaned)
    jd["locationMentions"] = dedupe(locs)[:10]

    # sections from HTML first, then text fallback for missing
    buckets_html = parse_sections_from_html(desc_html) if desc_html else {}
    buckets_text = split_sections_from_text(cleaned) if not buckets_html else {}
    resp = buckets_html.get("resp") or buckets_text.get("resp") or []
    req = buckets_html.get("req") or buckets_text.get("req") or []
    ben = buckets_html.get("ben") or buckets_text.get("ben") or []

    # prune headings that slipped in
    drop_heads = ("location:", "what you’ll do", "what you'll do", "why join", "about the company")
    resp = [x for x in resp if not x.lower().startswith(drop_heads)]
    req  = [x for x in req if not x.lower().startswith(drop_heads)]

    jd["responsibilitiesList"] = resp[:20]
    jd["benefitsRaw"] = ben[:20]

    # normalize benefits from full text as well
    ben_norm = set()
    tl = cleaned.lower()
    if "remote-first" in tl or "remote first" in tl or "remote" in tl: ben_norm.add("Remote")
    if "pto" in tl or "vacation" in tl or "holidays" in tl: ben_norm.add("PTO")
    if "parental leave" in tl or "maternity" in tl or "paternity" in tl: ben_norm.add("Parental Leave")
    if "stock" in tl or "equity" in tl or "options" in tl: ben_norm.add("Equity")
    if "health" in tl or "medical" in tl: ben_norm.add("Health")
    if "dental" in tl: ben_norm.add("Dental")
    if "vision" in tl: ben_norm.add("Vision")
    jd["benefitsNormalized"] = sorted(ben_norm)

    # years / seniority
    y_min, y_max = parse_years_range(cleaned)
    jd["yearsExperienceRequired"] = y_min if y_min is not None else y_max
    jd["seniorityLevel"] = adjust_seniority(sen, y_min, y_max, job.get("title") or "") or (sen or None)

    # salary with guardrails
    sal_min, sal_max, sal_cur, sal_period, sal_text, ctx_mismatch = parse_salary(cleaned)
    if any([sal_min, sal_max, sal_cur, sal_period]):
        jd["compensationText"] = sal_text or ""
        jd["compensationParsed"] = {"min": sal_min, "max": sal_max, "currencySymbol": sal_cur, "period": sal_period}

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

    # domain/product
    domain = []
    if "healthcare" in tl or "hipaa" in tl or "hitrust" in tl: domain.append("Healthcare")
    if "energy" in tl: domain.append("Energy")
    if "blockchain" in tl or "crypto" in tl: domain.append("Crypto/Blockchain")
    if "database" in tl or "sql" in tl: domain.append("Databases")
    jd["industryDomain"] = sorted(list(set(domain)))

    # compliance
    jd["complianceRegulations"] = parse_compliance(cleaned)

    # work hours / on-call
    wh = parse_work_hours(cleaned)
    if wh: jd["workHoursWindow"] = wh
    oc = parse_on_call(cleaned)
    if oc: jd["workSchedule"] = oc

    # DEI / culture / process
    jd["deiStatementPresence"] = True if re.search(r"equal opportunity|diverse|diversity|inclusive", cleaned, re.I) else False
    if "founder-led" in tl: jd["cultureSignals"].append("founder-led")
    if "remote-first" in tl: jd["cultureSignals"].append("remote-first")
    jd["cultureSignals"] = sorted(list(set(jd["cultureSignals"])))
    jd["applicationProcess"] = ["apply via platform"] if re.search(r"\bapply\b", cleaned, re.I) else []
    jd["workEnvironment"] = ["remote"] if jd["remoteMode"] in ("remote","hybrid") else []

    # keywords
    jd["keywordsTaxonomy"] = keywords_taxonomy(cleaned)

    # ---------------- LLM micro-call (if needed) ----------------
    need_keys = []
    if len(jd["responsibilitiesList"]) < 3: need_keys.append("responsibilitiesList")
    if len(req) < 3: need_keys.append("requirements")
    if not jd.get("workHoursWindow") and re.search(r"\boverlap\b", cleaned, re.I): need_keys.append("workHoursWindow")
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

    # merge LLM outputs conservatively
    if isinstance(llm_added.get("responsibilitiesList"), list) and len(jd["responsibilitiesList"]) < 3:
        jd["responsibilitiesList"] = dedupe(jd["responsibilitiesList"] + llm_added["responsibilitiesList"])[:20]
    if isinstance(llm_added.get("requirements"), list) and len(req) < 3:
        req = llm_added["requirements"][:20]
    jd["requirementsList"] = req
    if not jd.get("workHoursWindow") and isinstance(llm_added.get("workHoursWindow"), str):
        jd["workHoursWindow"] = llm_added["workHoursWindow"][:100]
    if not jd.get("outcomesKpis") and isinstance(llm_added.get("outcomesKpis"), list):
        jd["outcomesKpis"] = llm_added["outcomesKpis"][:10]
    summary = (llm_added.get("summary_tldr") or "").strip()

    # QA
    flags, score = qa_flags_and_score(jd, {"salary_ctx_mismatch": sal_text and ctx_mismatch, "cleaned_text": cleaned})

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
    if len(raw_html_trunc) > 2000:
        raw_html_trunc = raw_html_trunc[:2000] + "\n...[truncated]..."

    return {
        "jd_v3": jd,
        "jd_v3_meta": meta,
        "jd_v3_qa": {"score": score, "flags": flags},
        "legacy": legacy,
        "cleaned_text": cleaned,
        "raw_html_trunc": raw_html_trunc
    }

# --------------------------- Windowing & selection ---------------------------

def next_window_number() -> int:
    doc = jobs_col.find_one(
        {"jd_v3_eval_window": {"$exists": True}},
        sort=[("jd_v3_eval_window", -1)],
        projection={"jd_v3_eval_window": 1, "_id": 0}
    )
    return int(doc["jd_v3_eval_window"]) + 1 if doc else 1

def assign_window(window_size: int) -> Tuple[int, int]:
    wn = next_window_number()
    cur = jobs_col.find(
        {"jd_v3_eval_window": {"$exists": False}},
        projection={"_id":1, "last_seen_at":1, "apply_url":1},
        sort=[("last_seen_at", 1), ("_id", 1)],
        limit=window_size
    )
    jobs = list(cur)
    if not jobs: return (wn, 0)
    ops = []
    for idx, j in enumerate(jobs):
        split = "train" if idx < int(0.70*window_size) else ("val" if idx < int(0.85*window_size) else "test")
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
        "description_html":1, "tags":1, "last_seen_at":1, "jd_v3_meta":1, "jd_v3_eval_index":1, "jd_v3_eval_split":1
    }
    return list(jobs_col.find(q, projection=proj, sort=[("jd_v3_eval_index", 1)]))

def find_prior_failures() -> List[Dict[str, Any]]:
    q = {"jd_v3_meta.status": {"$in": ["error","timeout"]}}
    proj = {"_id":1, "title":1, "raw":1, "employment_type":1, "description_html":1, "tags":1, "jd_v3_meta":1}
    return list(jobs_col.find(q, projection=proj, sort=[("jd_v3_meta.enrichedAt", 1)], limit=200))

# --------------------------- Printing ---------------------------

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
    jd = enriched["jd_v3"]
    print("\n--- ENRICHED FIELDS (jd_v3) ---")
    def p(label, val):
        if isinstance(val, (list, tuple)):
            print(f"{label}: " + ("; ".join([json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x) for x in val]) if val else "[]"))
        elif isinstance(val, dict):
            print(f"{label}: {json.dumps(val, ensure_ascii=False)}")
        else:
            print(f"{label}: {val if val not in (None, '') else 'null'}")
    if PRINT_FIELDS:
        for key in PRINT_FIELDS:
            p(key, jd.get(key))
    else:
        for key in jd.keys():
            p(key, jd.get(key))
    print("-"*80)

# --------------------------- Progress UI ---------------------------

def render_progress(done: int, total: int, width: int = 28) -> str:
    if total <= 0: return f"{done} / ?"
    frac = min(1.0, done/total)
    filled = int(round(frac * width))
    bar = "█" * filled + " " * (width - filled)
    return f"{bar}  {done:,} / {total:,} ({frac*100:4.1f}%)"

# --------------------------- Ask-on-error ---------------------------

def timed_input(prompt: str, timeout: int = 300) -> Optional[str]:
    if platform.system().lower().startswith("win"):
        print("[ASK] Windows timed input not supported; auto-skip.")
        return None
    result = [None]
    def _get():
        try: result[0] = input(prompt)
        except Exception: result[0] = None
    th = threading.Thread(target=_get, daemon=True)
    th.start()
    th.join(timeout)
    return result[0]

# --------------------------- Processing ---------------------------

def process_jobs(docs: List[Dict[str, Any]], totals: Dict[str, int]) -> None:
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENCY) as ex:
        fut = {ex.submit(enrich_one, d): d for d in docs}
        for f in as_completed(fut):
            job = fut[f]
            try:
                out = f.result()
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
                if PRINT_ALL or PRINT_FIELDS or PRINT_HTML or PRINT_RAW_HTML:
                    print_success_artifacts(job, out)
            except Exception as e:
                msg = str(e)
                status = "timeout" if "timeout" in msg.lower() else "error"
                jobs_col.update_one({"_id": job["_id"]}, {"$set": {
                    "jd_v3_meta.status": status,
                    "jd_v3_meta.last_error": msg,
                    "jd_v3_meta.enrichedAt": now_iso()
                }}))
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
                            }}})

# --------------------------- Main ---------------------------

def main():
    totals = {"success":0, "error":0, "timeout":0}

    if ARGS.retry_failures:
        prev = find_prior_failures()
        if prev:
            total_prev = len(prev); processed_prev = 0
            for i in range(0, total_prev, BATCH_SIZE):
                chunk = prev[i:i+BATCH_SIZE]
                process_jobs(chunk, totals)
                processed_prev += len(chunk)
                if not ARGS.no_progress:
                    sys.stdout.write("\r[Retry failures] " + render_progress(processed_prev, total_prev))
                    sys.stdout.flush()
            if not ARGS.no_progress:
                sys.stdout.write("\n")

    if ARGS.job_id:
        doc = jobs_col.find_one({"_id": ARGS.job_id})
        if not doc:
            print(json.dumps({"ok": False, "error": "job not found", "id": ARGS.job_id}, indent=2))
            return
        process_jobs([doc], totals)
        print(json.dumps({"ok": True, "processed": 1, "window_assigned": 0, "success": totals["success"],
                          "error": totals["error"], "timeout": totals["timeout"]}, indent=2))
        return

    wn, assigned = assign_window(ARGS.window_size)
    if assigned == 0:
        print(json.dumps({"ok": True, "processed": 0, "window_assigned": 0, "message": "No unassigned jobs found."}, indent=2))
        return

    docs = find_jobs_for_window(wn)
    total = len(docs); processed = 0

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
