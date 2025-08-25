# resume_structurer.py
import os, re, json, time, argparse
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

# IO / parsing
import pdfplumber
import pytesseract
import cv2
import numpy as np
from PIL import Image
from docx import Document
from odf.opendocument import load
from odf import text, table
from odf.element import Element

# utils
import unicodedata
import phonenumbers
from langdetect import detect as _ld_detect
from dateutil import parser as dateparser
from rapidfuzz import fuzz

# ----------------------------
# CONFIG
# ----------------------------
LLAMA_API_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "phi-3.5-mini-instruct-q5_k_s"
TEMPERATURE = 0.0
MAX_TOKENS = 1200
REQUEST_CONNECT_TIMEOUT = 5
REQUEST_READ_TIMEOUT = 300

# Tesseract path (adjust if needed)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Optional Ghostscript on PATH (safe no-op if missing)
gs_path = r"C:\Program Files\gs\gs10.05.1\bin"
if os.path.isdir(gs_path) and gs_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = gs_path + os.pathsep + os.environ.get("PATH", "")

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRY_LLM_ON_NONRESUME = True  # still ask LLM to structure whatever it can


# ----------------------------
# Errors
# ----------------------------
class LLMError(Exception):
    pass

# ----------------------------
# Generic text normalization
# ----------------------------
def normalize_text_generic(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).replace("\u00ad", "")
    # collapse weird header spacing: "A M A N" -> "AMAN"
    t = re.sub(r"(?:\b[A-Z]\s)+(?:[A-Z]\b)", lambda m: m.group(0).replace(" ", ""), t)
    # collapse spaces
    t = re.sub(r"[ \t]+", " ", t)
    # join wrapped lines (prev ends with word/comma/semicolon; next starts lowercase)
    lines, out = [l.rstrip() for l in t.splitlines()], []
    for l in lines:
        if out:
            prev = out[-1]
            if re.search(r"[A-Za-z0-9,;]$", prev) and l[:1].islower():
                out[-1] = prev + " " + l.lstrip()
                continue
        out.append(l)
    t = "\n".join(out)
    return re.sub(r"\n{3,}", "\n\n", t).strip()

# ----------------------------
# Headings & segmentation
# ----------------------------
CANON_HINTS = {
    "experience": ["experience","work","employment","career","beruf","erfahrung","praxis"],
    "education": ["education","academic","studies","ausbildung","bildung","studium","school"],
    "skills": ["skills","kenntnisse","fähigkeiten","competencies","it","computer"],
    "languages": ["languages","sprachen","idiomas","langues"],
    "projects": ["projects","projekt"],
    "certifications": ["certifications","certificates","zertifikat","zertifikate"],
    "profile": ["profile","summary","objective","about","über mich"],
    "awards": ["awards","honors","preise"],
    "publications": ["publications","publikationen"],
    "volunteering": ["volunteer","ehrenamt","community"],
    "references": ["references","referenzen","referees"],
}

def is_heading(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    wc = len(s.split())
    if wc <= 8 and s.endswith(":"):
        return True
    letters = re.findall(r"[A-Za-z]", s)
    if letters:
        caps = sum(1 for ch in letters if ch.isupper())
        if caps / len(letters) >= 0.65 and wc <= 8:
            return True
    # vocabulary-based heading (no colon / not all caps)
    s_l = s.lower().strip(": ")
    for canon, hints in CANON_HINTS.items():
        if any(h == s_l for h in hints+[canon]):
            return True
    return False

def map_to_canonical(label: str) -> str:
    label_l = (label or "").lower().strip(" :")
    best, score = "other", 0
    for canon, hints in CANON_HINTS.items():
        for h in hints + [canon]:
            sc = max(
                fuzz.partial_ratio(label_l, h),
                80 if (h in label_l or label_l in h) else 0,
                100 if label_l == h else 0
            )
            if sc > score:
                score, best = sc, canon
    return best if score >= 70 else "other"

def segment_by_shape(text: str) -> Dict[str, str]:
    segs: Dict[str, List[str]] = {"profile": []}
    current = "profile"
    for line in (text or "").splitlines():
        if is_heading(line):
            current = map_to_canonical(re.sub(r":\s*$","",line.strip()))
            segs.setdefault(current, [])
            continue
        segs[current].append(line)
    return {k: "\n".join(v).strip() for k, v in segs.items() if any(x.strip() for x in v)}

# ----------------------------
# Contacts & language
# ----------------------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_CAND_RE = re.compile(r"(?:\+?\d[\d\s().\-]{6,}\d)")

def extract_contacts(text: str, default_region="AT") -> dict:
    emails = sorted(set(EMAIL_RE.findall(text or "")))

    # collect candidates; filter out MRZ-like & absurd lengths
    raw = set(PHONE_CAND_RE.findall(text or ""))
    cleaned = []
    for s in raw:
        s2 = s.strip()
        if "<" in s2 or len(re.sub(r"\D", "", s2)) > 16:
            continue
        try:
            for m in phonenumbers.PhoneNumberMatcher(s2, default_region):
                e164 = phonenumbers.format_number(m.number, phonenumbers.PhoneNumberFormat.E164)
                if 10 <= len(re.sub(r"\D","", e164)) <= 15:
                    cleaned.append(e164)
        except Exception:
            pass
    phones = sorted(set(cleaned))
    return {"emails": emails, "phones": phones}

def detect_language(text: str) -> str:
    try:
        return _ld_detect(text or "")
    except Exception:
        return "unknown"

# ----------------------------
# Date normalization
# ----------------------------
def normalize_date(s: str) -> str:
    if not isinstance(s, str) or not s.strip():
        return ""
    s = s.strip().replace("–","-").replace("—","-")
    s = re.sub(r"\bto\b", "-", s, flags=re.I)
    if re.fullmatch(r"(?i)(now|present|current)", s):
        return "present"
    # fast paths
    m = re.fullmatch(r"(19|20)\d{2}", s)
    if m:
        return m.group(0)
    try:
        # use a stable default to avoid "today" leaking in
        dt = dateparser.parse(s, default=datetime(2000,1,1))
        if dt:
            return dt.strftime("%Y-%m")
    except Exception:
        pass
    return s

# ----------------------------
# OCR helpers
# ----------------------------
def preprocess_image_for_ocr(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    # light denoise
    den = cv2.fastNlMeansDenoising(gray, None, 20, 7, 21)
    # adaptive binarization
    th = cv2.adaptiveThreshold(den, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 31, 7)
    # slight blur to connect thin strokes
    bl = cv2.GaussianBlur(th, (3, 3), 0)
    # upscale
    up = cv2.resize(bl, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return up

def extract_text_from_pdf(pdf_path: str) -> str:
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            txt = (page.extract_text() or "").strip()
            if txt:
                out.append(txt)
                continue
            # OCR fallback
            pil_im = page.to_image(resolution=300).original  # PIL RGB
            im_rgb = np.array(pil_im)  # RGB
            processed = preprocess_image_for_ocr(im_rgb)
            ocr = pytesseract.image_to_string(processed, config="--oem 3 --psm 6")
            if ocr.strip():
                out.append(ocr)
    return "\n".join(out).strip()

def extract_text_from_docx(docx_path: str) -> str:
    doc = Document(docx_path)
    return "\n".join(p.text for p in doc.paragraphs).strip()

def _get_text_recursive_odf(element) -> str:
    text_content = ""
    for node in element.childNodes:
        if node.nodeType == node.TEXT_NODE:
            text_content += node.data
        elif isinstance(node, Element):
            text_content += _get_text_recursive_odf(node)
    return text_content

def extract_text_from_odt(odt_path: str) -> str:
    doc = load(odt_path)
    all_text: List[str] = []
    for elem in doc.getElementsByType(text.P) + doc.getElementsByType(text.H) + doc.getElementsByType(text.Span):
        para_text = _get_text_recursive_odf(elem).strip()
        if para_text:
            all_text.append(para_text)
    for table_elem in doc.getElementsByType(table.Table):
        for row in table_elem.getElementsByType(table.TableRow):
            row_text = []
            for cell in row.getElementsByType(table.TableCell):
                for p in cell.getElementsByType(text.P):
                    cell_text = _get_text_recursive_odf(p).strip()
                    if cell_text:
                        row_text.append(cell_text)
            if row_text:
                all_text.append(" | ".join(row_text))
    return "\n".join(all_text).strip()

def extract_text_from_image(image_path: str) -> str:
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, config="--oem 3 --psm 6").strip()

def process_image_file(image_path: str) -> str:
    bgr = cv2.imread(image_path)
    if bgr is None:
        return ""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pre = preprocess_image_for_ocr(rgb)
    return pytesseract.image_to_string(pre, config="--oem 3 --psm 6").strip()

# ----------------------------
# Quality & classification
# ----------------------------
def is_low_quality_text(text: str, min_length=300, min_alpha_ratio=0.65,
                        max_symbol_ratio=0.28, max_garbage_lines=0.35) -> bool:
    lines = (text or "").splitlines()
    if not lines:
        return True
    total_chars = total_alpha = total_symbols = 0
    garbage_lines = 0
    for line in lines:
        s = line.strip()
        total_chars += len(s)
        total_alpha += sum(c.isalpha() for c in s)
        total_symbols += sum(1 for c in s if not c.isalnum() and not c.isspace())
        if len(s) < 5 or re.fullmatch(r'[^a-zA-Z0-9]+', s or ""):
            garbage_lines += 1
    if total_chars == 0:
        return True
    alpha_ratio = total_alpha / max(total_chars, 1)
    symbol_ratio = total_symbols / max(total_chars, 1)
    garbage_ratio = garbage_lines / max(len(lines), 1)
    return (total_chars < min_length or
            alpha_ratio < min_alpha_ratio or
            symbol_ratio > max_symbol_ratio or
            garbage_ratio > max_garbage_lines)

MRZ_RE = re.compile(r"^[A-Z0-9<]{20,}$")
PASSPORT_HINTS = re.compile(r"(?i)\bpassport\b|P<|machine[- ]readable", re.I)
CERT_HINTS = re.compile(r"(?i)(certificate|trade test|board of|education board|marks?|license|licen[cs]e|driving)", re.I)

def classify_document(text: str, filename: str = "") -> str:
    """Heuristic doc classifier with strong passport MRZ detection. Fail-safe bias: 'resume'."""
    t = (text or "").strip()
    if not t:
        return "other"

    fname = (filename or "").lower()
    if "passport" in fname:
        # filename says passport → trust unless very strong resume signal
        filename_passport_hint = True
    else:
        filename_passport_hint = False

    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    # MRZ: 2 consecutive lines of A–Z0–9<, length 30–44, with at least one '<<'
    def is_mrz_line(s: str) -> bool:
        if not (30 <= len(s) <= 44):
            return False
        if not re.fullmatch(r"[A-Z0-9<]+", s):
            return False
        return "<<" in s

    mrz_idx = [i for i, ln in enumerate(lines) if is_mrz_line(ln)]
    mrz_pair = any((j + 1 in mrz_idx) for j in mrz_idx)  # consecutive pair

    if mrz_pair or (filename_passport_hint and _contains_any(t.lower(), PASSPORT_KEYWORDS) >= 1):
        return "passport"

    # Resume vs certificate signals
    cert_score = _contains_any(t.lower(), CERT_KEYWORDS)
    resume_score = _contains_any(t.lower(), RESUME_KEYWORDS)
    bullet_count = sum(1 for ln in lines if BULLET_RE.match(ln))
    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", t))
    low_quality = is_low_quality_text(t, min_length=250)

    # Strong cert
    if cert_score >= 3 and resume_score <= 1 and bullet_count <= 1:
        return "certificate"
    # Strong resume
    if resume_score >= 2 or bullet_count >= 3 or year_hits >= 4:
        return "resume"
    # Low-quality but certificate-ish
    if low_quality and cert_score >= 2 and resume_score == 0:
        return "certificate"

    # Default bias: try to parse as resume
    return "resume"

# ----------------------------
# Chunking & merge
# ----------------------------
def chunk_text_by_chars(text: str, max_chars: int = 6000, overlap: int = 400):
    text = (text or "").strip()
    if len(text) <= max_chars:
        return [text] if text else []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(start + max_chars, n)
        nl = text.rfind("\n", start, end)
        if nl != -1 and nl > start + 1000:
            end = nl
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end)
    return [c for c in chunks if c]

def merge_structured_json(a: dict, b: dict) -> dict:
    import json as _json
    out = _json.loads(_json.dumps(a))  # deep copy

    for k in ["full_name","job_title","summary","availability"]:
        va, vb = out.get(k, ""), b.get(k, "")
        out[k] = vb if (vb and len(vb) > len(va)) else va

    out.setdefault("contact", {})
    for key in ["email","emails","phones","address"]:
        va = out["contact"].get(key)
        vb = b.get("contact", {}).get(key)
        if isinstance(va, list) or isinstance(vb, list):
            sa = set(va or [])
            sb = set(vb or [])
            out["contact"][key] = sorted(sa | sb)
        else:
            out["contact"][key] = vb or va or ([] if key.endswith("s") else "")

    def merge_list(sec, keys):
        out.setdefault(sec, [])
        seen = set()
        for rec in out[sec] + b.get(sec, []):
            t = tuple((rec.get(k, "") if isinstance(rec.get(k), str) else str(rec.get(k))).strip().lower() for k in keys)
            if t not in seen:
                seen.add(t); out[sec].append(rec)
    merge_list("experience", ("company","title","start","end","location"))
    merge_list("education", ("institution","degree","start","end","location"))
    merge_list("projects", ("name","role","start","end"))
    merge_list("certifications", ("name","issuer","date"))
    merge_list("publications", ("title","venue","date"))
    merge_list("languages", ("language","level"))

    for k in ["volunteering","awards","clearances","keywords"]:
        va = out.get(k, []); vb = b.get(k, [])
        out[k] = sorted(set((va or []) + (vb or [])))

    out.setdefault("skills", {})
    for bucket in ["hard","soft","tools","domains"]:
        sa = set(out["skills"].get(bucket, []) or [])
        sb = set(b.get("skills", {}).get(bucket, []) or [])
        merged = sorted(sa | sb)
        if merged: out["skills"][bucket] = merged

    out["preferences"] = {**out.get("preferences", {}), **b.get("preferences", {})}
    out["meta"] = {**out.get("meta", {}), **b.get("meta", {})}
    return out

# ----------------------------
# Schema & parsing
# ----------------------------
from pydantic import BaseModel, Field, ValidationError

class ExperienceItem(BaseModel):
    company: str = ""
    title: str = ""
    start: str = ""
    end: str = ""
    location: str = ""
    bullets: List[str] = Field(default_factory=list)

class EducationItem(BaseModel):
    institution: str = ""
    degree: str = ""
    field_of_study: str = ""
    start: str = ""
    end: str = ""
    location: str = ""

class ProjectItem(BaseModel):
    name: str = ""
    role: str = ""
    start: str = ""
    end: str = ""
    tech: List[str] = Field(default_factory=list)
    description: str = ""

class CertificateItem(BaseModel):
    name: str = ""
    issuer: str = ""
    date: str = ""

class LanguageItem(BaseModel):
    language: str = ""
    level: str = ""

class PublicationItem(BaseModel):
    title: str = ""
    venue: str = ""
    date: str = ""
    link: str = ""

class SocialLink(BaseModel):
    platform: str = ""
    url: str = ""

class ResumeJSON(BaseModel):
    full_name: str = ""
    job_title: str = ""
    contact: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    keywords: List[str] = Field(default_factory=list)
    skills: Dict[str, List[str]] = Field(default_factory=dict)
    experience: List[ExperienceItem] = Field(default_factory=list)
    education: List[EducationItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    certifications: List[CertificateItem] = Field(default_factory=list)
    languages: List[LanguageItem] = Field(default_factory=list)
    publications: List[PublicationItem] = Field(default_factory=list)
    volunteering: List[str] = Field(default_factory=list)
    awards: List[str] = Field(default_factory=list)
    links: List[SocialLink] = Field(default_factory=list)
    clearances: List[str] = Field(default_factory=list)
    preferences: Dict[str, Any] = Field(default_factory=dict)
    availability: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)

JSON_INSTRUCTIONS = """
You are a structured resume extractor. Return ONLY valid JSON for the schema.
- No commentary/markdown.
- Fill as many fields as possible; unknowns = empty string/array.
- company = employer; title = role (do not swap).
- Normalize dates to 'YYYY-MM' (or 'YYYY'); use 'present' only for ongoing jobs.
- Use short, factual bullet points.
"""

COMPACT_SCHEMA = {
  "full_name": "", "job_title": "",
  "contact": {"emails": [], "phones": [], "address": ""},
  "summary": "", "keywords": [],
  "skills": {"hard": [], "soft": [], "tools": [], "domains": []},
  "experience": [{"company": "", "title": "", "start": "", "end": "", "location": "", "bullets": []}],
  "education": [{"institution": "", "degree": "", "field_of_study": "", "start": "", "end": "", "location": ""}],
  "projects": [{"name": "", "role": "", "start": "", "end": "", "tech": [], "description": ""}],
  "certifications": [{"name": "", "issuer": "", "date": ""}],
  "languages": [{"language": "", "level": ""}],
  "publications": [{"title": "", "venue": "", "date": "", "link": ""}],
  "volunteering": [], "awards": [], "links": [{"platform": "", "url": ""}],
  "clearances": [], "preferences": {"relocation": "", "remote": "", "travel": "", "salary": "", "locations": []},
  "availability": "", "meta": {}
}

def extract_experience_candidates(text: str) -> str:
    """Pull likely experience lines to help LLM on messy CVs (e.g., Aman)."""
    lines = [l for l in (text or "").splitlines() if l.strip()]
    keep = []
    DATE_PAT = re.compile(r"((19|20)\d{2})(\s*[–—\-]\s*(present|(19|20)\d{2}))?", re.I)
    ROLE_HINTS = re.compile(r"(manager|developer|designer|engineer|teacher|baker|courier|assistant|verkäufer|mechanic|driver|analyst)", re.I)
    for i, l in enumerate(lines):
        if DATE_PAT.search(l) or ROLE_HINTS.search(l):
            keep.append(l.strip())
            # also pull the next line if it looks like bullets/details
            if i+1 < len(lines) and len(lines[i+1].strip()) > 15:
                keep.append(lines[i+1].strip())
    return "\n".join(keep[:3000])

def build_messages(resume_text: str, filename: str, segments: dict = None):
    seg_dump = json.dumps(segments or {"_all": resume_text}, ensure_ascii=False)
    hints = extract_experience_candidates(resume_text)
    return [
        {"role":"system","content":"Return ONLY valid JSON. Output must start with '{' and end with '}'. Unknowns = empty strings/arrays. No markdown."},
        {"role":"user","content":(
            JSON_INSTRUCTIONS.strip()
            + "\n\nSCHEMA (compact):\n" + json.dumps(COMPACT_SCHEMA, ensure_ascii=False)
            + "\n\nTEXT SEGMENTS:\n" + seg_dump
            + "\n\nHINTS (candidate experience lines):\n" + hints[:4000]
            + "\n\nFILENAME: " + filename
        )}
    ]

# ----------------------------
# Document type classifier
# ----------------------------
MRZ_RE = re.compile(r"^[A-Z0-9<]{20,}$")  # passport MRZ-like lines
BULLET_RE = re.compile(r"^[•\-\u2022]\s+")  # bullets

CERT_KEYWORDS = [
    "board of technical education", "school education board", "marks sheet", "trade certificate",
    "iti", "industrial training", "grade", "roll no", "regn no", "certificate no", "vocational",
    "national trade certificate", "apprenticeship", "competency certificate"
]
PASSPORT_KEYWORDS = [
    "passport", "republic of", "authority", "date of issue", "date of expiry",
    "place of birth", "nationality"
]
RESUME_KEYWORDS = [
    "work experience", "experience", "employment", "education", "skills", "projects",
    "certifications", "summary", "objective", "languages", "profile", "responsibilities",
    "achievements", "professional", "curriculum vitae", "resume"
]

def _contains_any(text: str, words: list) -> int:
    t = text.lower()
    return sum(1 for w in words if w in t)

# ----------------------------
# ID / Passport extraction (deterministic)
# ----------------------------
MRZ_LINE_RE = re.compile(r"^[A-Z0-9<]{30,50}$")
DOB_RE      = re.compile(r"\b(?:date of birth|dob)\s*[:\- ]\s*([0-9]{1,2}[\/\.\-][0-9]{1,2}[\/\.\-][0-9]{2,4})", re.I)
DOE_RE      = re.compile(r"\b(?:date of expiry|expiry|expires)\s*[:\- ]\s*([0-9]{1,2}[\/\.\-][0-9]{1,2}[\/\.\-][0-9]{2,4})", re.I)
DOCNO_RE    = re.compile(r"\b(passport|document)\s*(no|number)?\s*[:\- ]\s*([A-Z0-9]{6,15})", re.I)
NAT_RE      = re.compile(r"\b(nationality)\s*[:\- ]\s*([A-Za-z ]{2,})", re.I)
NAME_RE     = re.compile(r"\b(surname|last name|family name|name)\s*[:\- ]\s*([A-Za-z ,.'\-]{2,})", re.I)
GIVEN_RE    = re.compile(r"\b(given names?|first name)\s*[:\- ]\s*([A-Za-z ,.'\-]{2,})", re.I)

def _safe_dt(s: str) -> str:
    try:
        dt = dateparser.parse(s, dayfirst=True) or dateparser.parse(s, dayfirst=False)
        return dt.strftime("%Y-%m-%d") if dt else ""
    except Exception:
        return ""

def _parse_mrz(lines: list) -> dict:
    """
    Basic TD3 two-line MRZ parser:
    P<ISSUERNAME<<GIVEN<NAMES<<<<<<<<<<<<<<<<<<<
    NNNNNNNNNNISSUERYYMMDDMYYMMDD<<<<<<<<<<<<<NN
    We’ll grab: document_number, issuing_state, nationality?, surname, given_names, dob, sex, expiry
    """
    mrz = [ln.strip() for ln in lines if MRZ_LINE_RE.match(ln)]
    if len(mrz) < 2:
        return {}

    l1, l2 = mrz[0], mrz[1]
    out = {}

    # Line 1: P<XXXNAME<<GIVEN<NAMES
    # First char is 'P', then '<', then issuing_state (3 letters)
    if len(l1) >= 5 and l1[0] in ("P", "A", "C"):  # passport types
        issuing = l1[2:5].replace("<", "").strip()
        out["issuing_state"] = issuing or ""

        # Name field after position 5: SURNAME<<GIVEN<NAMES
        name_field = l1[5:].strip("<")
        parts = name_field.split("<<", 1)
        surname = parts[0].replace("<", " ").strip() if parts else ""
        given   = parts[1].replace("<", " ").strip() if len(parts) > 1 else ""
        out["surname"] = surname
        out["given_names"] = given

    # Line 2: document number (9 chars) + nationality (3) + dob (6) + sex (1) + expiry (6)
    if len(l2) >= 30:
        docno = l2[0:9].replace("<", "").strip()
        nat   = l2[10:13].replace("<", "").strip()
        dob   = l2[13:19]
        sex   = l2[20:21]
        doe   = l2[21:27]
        # Normalize dates (YYMMDD)
        def yymmdd(x):
            if not re.fullmatch(r"\d{6}", x or ""): return ""
            yy, mm, dd = x[0:2], x[2:4], x[4:6]
            # naive century heuristic: 00-24 -> 2000s, else 1900s
            century = "20" if int(yy) <= 24 else "19"
            return f"{century}{yy}-{mm}-{dd}"
        out.update({
            "document_number": docno,
            "nationality": nat,
            "date_of_birth": yymmdd(dob),
            "sex": {"M":"M","F":"F"}.get(sex, ""),
            "date_of_expiry": yymmdd(doe),
        })
    return {k:v for k,v in out.items() if v}

def extract_id_document_fields(text: str) -> dict:
    """Try MRZ first; fall back to loose regex for common fields."""
    lines = [ln.strip() for ln in (text or "").splitlines() if ln.strip()]
    mrz = _parse_mrz(lines)
    if mrz:
        return {"source": "mrz", **mrz}

    t = text or ""
    d = {}
    m = DOCNO_RE.search(t);      d["document_number"] = (m.group(3) if m else "") or d.get("document_number","")
    m = DOB_RE.search(t);        d["date_of_birth"]   = _safe_dt(m.group(1)) if m else d.get("date_of_birth","")
    m = DOE_RE.search(t);        d["date_of_expiry"]  = _safe_dt(m.group(1)) if m else d.get("date_of_expiry","")
    m = NAT_RE.search(t);        d["nationality"]     = (m.group(2).strip() if m else "") or d.get("nationality","")
    m = NAME_RE.search(t);       d["surname"]         = (m.group(2).strip() if m else "") or d.get("surname","")
    m = GIVEN_RE.search(t);      d["given_names"]     = (m.group(2).strip() if m else "") or d.get("given_names","")
    d["source"] = d.get("source","regex")
    # drop empties
    return {k:v for k,v in d.items() if v}





def classify_document(text: str, filename: str = "") -> str:
    """Heuristic doc classifier. Fail-safe: prefer 'resume' unless strong signals suggest otherwise."""
    t = (text or "").strip()
    if not t:
        return "other"

    # Passport: MRZ or strong passport cues
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    mrz_hits = sum(1 for ln in lines if MRZ_RE.match(ln) and "<<" in ln)
    if mrz_hits >= 1 or _contains_any(t, PASSPORT_KEYWORDS) >= 2:
        return "passport"

    # Certificates / mark sheets / training cards
    cert_score = _contains_any(t, CERT_KEYWORDS)
    # Penalize if resume sections present
    resume_score = _contains_any(t, RESUME_KEYWORDS)
    bullet_count = sum(1 for ln in lines if BULLET_RE.match(ln))
    year_hits = len(re.findall(r"\b(19|20)\d{2}\b", t))

    # Very OCR-y documents often have lots of symbols and almost no alphabetic density
    low_quality = is_low_quality_text(t, min_length=250)

    # Strong certificate cues with weak resume cues => certificate
    if cert_score >= 3 and resume_score <= 1 and bullet_count <= 1:
        return "certificate"

    # If we get here and see typical resume structure, call it a resume
    if resume_score >= 2 or bullet_count >= 3 or year_hits >= 4:
        return "resume"

    # Low quality + certificate-ish (e.g., scanned mark sheets)
    if low_quality and cert_score >= 2 and resume_score == 0:
        return "certificate"

    # Fall back: when unsure, prefer RESUME so we at least try LLM
    return "resume"


# ----------------------------
# LLM call & JSON repair
# ----------------------------
def call_llm(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local"}
    try:
        print(f"[LLM] POST {LLAMA_API_URL}  (connect {REQUEST_CONNECT_TIMEOUT}s, read {REQUEST_READ_TIMEOUT}s)", flush=True)
        resp = __import__("requests").post(
            LLAMA_API_URL, json=payload, headers=headers,
            timeout=(REQUEST_CONNECT_TIMEOUT, REQUEST_READ_TIMEOUT),
        )
        print(f"[LLM] HTTP {resp.status_code}", flush=True)
    except Exception as e:
        raise LLMError(f"Connection error to local LLM: {e}")
    if resp.status_code != 200:
        raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")
    try:
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise LLMError(f"Unexpected LLM response: {e}\nBody head:\n{resp.text[:500]}")
    if not content or not content.strip():
        raise LLMError("Empty response from LLM.")
    return content.strip()

def _strip_trailing_commas(s: str) -> str:
    return re.sub(r',\s*([}\]])', r'\1', s)

def _auto_close_brackets(s: str) -> str:
    stack, out = [], []
    pairs = {'{': '}', '[': ']'}
    for ch in s:
        out.append(ch)
        if ch in pairs:
            stack.append(ch)
        elif ch in (']','}'):
            if stack and pairs[stack[-1]] == ch:
                stack.pop()
            else:
                out.pop()
    while stack:
        out.append(pairs[stack.pop()])
    return ''.join(out)

def parse_llm_json(raw: str) -> ResumeJSON:
    first, last = raw.find("{"), raw.rfind("}")
    candidate = raw if (first == -1 or last == -1) else raw[first:last+1]
    try:
        obj = json.loads(candidate)
        return ResumeJSON(**obj)
    except Exception:
        repaired = _auto_close_brackets(_strip_trailing_commas(candidate))
        try:
            obj = json.loads(repaired)
        except json.JSONDecodeError:
            for cut in range(len(repaired), max(len(repaired)-2000, 0), -50):
                try:
                    obj = json.loads(_auto_close_brackets(_strip_trailing_commas(repaired[:cut])))
                    break
                except Exception:
                    continue
            else:
                raise LLMError(f"Invalid JSON from LLM after repair. Raw head:\n{raw[:1000]}")
        try:
            return ResumeJSON(**obj)
        except ValidationError as ve:
            raise LLMError(f"JSON did not match schema: {ve}\nJSON:\n{json.dumps(obj, indent=2)[:1500]}")

# ----------------------------
# File processing
# ----------------------------
def process_file(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        print("Processing PDF…")
        text = extract_text_from_pdf(file_path)
    elif ext == ".docx":
        print("Processing DOCX…")
        text = extract_text_from_docx(file_path)
    elif ext == ".odt":
        print("Processing ODT…")
        text = extract_text_from_odt(file_path)
    elif ext in [".jpg", ".jpeg", ".png"]:
        print("Processing image…")
        text = process_image_file(file_path) or extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    if is_low_quality_text(text):
        print("⚠️  Warning: Extracted text looks low quality (scan or layout).")
    return text

# ----------------------------
# Post-process
# ----------------------------
def postprocess_structured(d: dict, contacts: dict, lang: str) -> dict:
    d = json.loads(json.dumps(d))  # deep copy
    d.setdefault("contact", {})
    if contacts.get("emails") and not d["contact"].get("emails"):
        d["contact"]["emails"] = contacts["emails"]
    if contacts.get("phones") and not d["contact"].get("phones"):
        d["contact"]["phones"] = contacts["phones"]

    JOB_WORDS = re.compile(r"(verkäufer|teacher|manager|developer|designer|mechanic|baker|courier|assistant|engineer|driver|analyst)", re.I)
    for rec in d.get("experience", []) or []:
        if isinstance(rec.get("start"), str): rec["start"] = normalize_date(rec["start"])
        if isinstance(rec.get("end"), str):   rec["end"]   = normalize_date(rec["end"])
        comp = (rec.get("company") or "").strip()
        title = (rec.get("title") or "").strip()
        if comp and title and JOB_WORDS.search(comp) and not JOB_WORDS.search(title):
            rec["company"], rec["title"] = title, comp
        rec["company"] = rec.get("company","").strip(" |-")
        rec["title"]   = rec.get("title","").strip(" |-")
        rec["location"]= rec.get("location","").replace("|","").strip()

    # Education: 'present' is rarely valid — clamp obviously wrong cases
    this_year = datetime.utcnow().year
    for rec in d.get("education", []) or []:
        if isinstance(rec.get("start"), str): rec["start"] = normalize_date(rec["start"])
        if isinstance(rec.get("end"), str):   rec["end"]   = normalize_date(rec["end"])
        if rec.get("end","").lower() == "present":
            # if start is a year long ago, clear 'present'
            try:
                yr = int((rec.get("start") or "")[:4])
                if yr and this_year - yr > 6:
                    rec["end"] = ""
            except Exception:
                rec["end"] = ""

    # de-dup keywords & skill buckets
    d["keywords"] = sorted(set(d.get("keywords") or []))
    d.setdefault("skills", {})
    for b in ["hard","soft","tools","domains"]:
        if isinstance(d["skills"].get(b), list):
            d["skills"][b] = sorted(set(x.strip() for x in d["skills"][b] if x and isinstance(x, str)))

    d.setdefault("meta", {})
    d["meta"]["language"] = lang
    return d

# ----------------------------
# Public helpers
# ----------------------------
def iter_supported_files(root: str):
    exts = {".pdf", ".docx", ".odt", ".png", ".jpg", ".jpeg"}
    for p, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.join(p, f)

def extract_structured_in_chunks(full_text: str, filename: str, build_messages_fn) -> dict:
    chunks = chunk_text_by_chars(full_text, max_chars=6000, overlap=400)
    merged = None
    for i, ch in enumerate(chunks or [full_text], 1):
        print(f"[CHUNK {i}/{len(chunks) if chunks else 1}] → LLM", flush=True)
        msgs = build_messages_fn(ch, filename, {"_chunk": ch})
        raw = call_llm(msgs)
        part = parse_llm_json(raw)
        d = part.model_dump() if hasattr(part, "model_dump") else part.dict()
        merged = d if merged is None else merge_structured_json(merged, d)
    return merged or {}

# ----------------------------
# CLI main
# ----------------------------
def main():
    print("=== Rovari Local Resume Structurer v2 ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Process a directory (recursive). If omitted, opens file picker.")
    parser.add_argument("--force-type", type=str, choices=["resume","certificate","passport","other"],
                        help="Override document classification.")
    args = parser.parse_args()
    print("argv:", " ".join(os.sys.argv), flush=True)

    if args.dir:
        root = os.path.abspath(args.dir)
        print(f"Batch mode: scanning {root}", flush=True)
        paths = list(iter_supported_files(root))
        print(f"Found {len(paths)} files:", flush=True)
        for fp in paths: print(" -", fp, flush=True)
        if not paths:
            print("No supported files found. Exiting.", flush=True)
            return
    else:
        fp = pick_file()
        if not fp:
            print("No file selected. Exiting."); return
        paths = [fp]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for file_path in paths:
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        base = Path(file_path).stem
        txt_out = OUTPUT_DIR / f"{base}.{ts}.txt"
        json_out = OUTPUT_DIR / f"{base}.{ts}.json"
        err_out = OUTPUT_DIR / f"{base}.{ts}.error.txt"

        try:
            print(f"\n=== Processing: {file_path}", flush=True)
            # 1) Extract + normalize
            text = process_file(file_path)
            text = normalize_text_generic(text)
            segments = segment_by_shape(text)
            contacts = extract_contacts(text)
            lang = detect_language(text)

            # 2) Save raw text
            txt_out.write_text(text, encoding="utf-8")
            print(f"✔ Saved extracted text → {txt_out}", flush=True)

            # 3) Classify
            doc_type = args.force_type or classify_document(text, os.path.basename(file_path))
            print(f"• Classified as '{doc_type}'", flush=True)

            # 4) Build result dict (always define d)
            d = {
                "full_name": "", "job_title": "",
                "contact": contacts, "summary": "",
                "keywords": [],
                "skills": {"hard": [], "soft": [], "tools": [], "domains": []},
                "experience": [], "education": [], "projects": [],
                "certifications": [], "languages": [], "publications": [],
                "volunteering": [], "awards": [], "links": [],
                "clearances": [], "preferences": {}, "availability": "",
                "meta": {
                    "language": lang,
                    "source_file": os.path.basename(file_path),
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "model": MODEL_NAME,
                    "doc_type": doc_type,
                }
            }

            # 5) Parse path depending on type
            if doc_type == "resume":
                try:
                    d2 = extract_structured_in_chunks(text, os.path.basename(file_path), build_messages)
                    d2 = postprocess_structured(d2, contacts, lang)
                    d = merge_structured_json(d, d2)
                except Exception as e:
                    err = f"Resume parsing failed, wrote minimal JSON. Error: {e}"
                    print(f"ℹ️  {err}", flush=True)
                    err_out.write_text(err, encoding="utf-8")
            else:
                # Deterministic ID-doc fields + optional LLM
                try:
                    id_doc = extract_id_document_fields(text)
                    if id_doc:
                        d["meta"]["id_document"] = id_doc
                except Exception as e:
                    print(f"ℹ️  ID extraction skipped: {e}", flush=True)

                if TRY_LLM_ON_NONRESUME:
                    try:
                        msgs = build_messages(text, os.path.basename(file_path), {"_all": text, "_doc_type": doc_type})
                        raw = call_llm(msgs)
                        part = parse_llm_json(raw)
                        part_d = part.model_dump() if hasattr(part, "model_dump") else part.dict()
                        d = merge_structured_json(d, part_d)
                    except Exception as e:
                        msg = f"LLM on non-resume failed softly: {e}"
                        print(f"ℹ️  {msg}", flush=True)
                        err_out.write_text(msg, encoding="utf-8")

            # 6) Save JSON (always)
            d.setdefault("meta", {})
            d["meta"].update({
                "source_file": os.path.basename(file_path),
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "model": MODEL_NAME,
                "doc_type": d["meta"].get("doc_type", doc_type),
            })
            json_out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✔ Saved structured JSON → {json_out}", flush=True)

        except Exception as e:
            # last-resort minimal JSON with error recorded
            err_text = f"Fatal error on {file_path}: {e}"
            print(f"✖ {err_text}", flush=True)
            try: err_out.write_text(err_text, encoding="utf-8")
            except Exception: pass
            minimal = {
                "full_name": "", "job_title": "",
                "contact": {"emails": [], "phones": [], "address": ""},
                "summary": "", "keywords": [],
                "skills": {"hard": [], "soft": [], "tools": [], "domains": []},
                "experience": [], "education": [], "projects": [],
                "certifications": [], "languages": [], "publications": [],
                "volunteering": [], "awards": [], "links": [],
                "clearances": [], "preferences": {}, "availability": "",
                "meta": {
                    "error": err_text,
                    "source_file": os.path.basename(file_path),
                    "generated_at": datetime.utcnow().isoformat() + "Z",
                    "model": MODEL_NAME,
                    "doc_type": "other",
                }
            }
            try:
                json_out.write_text(json.dumps(minimal, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"✔ Wrote minimal JSON with error → {json_out}", flush=True)
            except Exception:
                print("✖ Could not write minimal JSON file.", flush=True)

if __name__ == "__main__":
    main()
