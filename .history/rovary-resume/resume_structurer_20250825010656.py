import os
import re
import json
import time
import cv2
import numpy as np
import requests
import pdfplumber
import pytesseract
from PIL import Image
from docx import Document
from odf.opendocument import load
from odf import text, table
from odf.element import Element
from tkinter import Tk, filedialog
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import Dict, Any, List
from pydantic import BaseModel, ValidationError, Field
from datetime import datetime
from pathlib import Path
# add near top
import argparse

def iter_supported_files(root: str):
    exts = {".pdf", ".docx", ".odt", ".png", ".jpg", ".jpeg"}
    for p, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.join(p, f)

# REPLACE your main() with this:
def main():
    print("=== Rovari Local Resume Structurer v2 ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Process all resumes in this directory (recursive).")
    args = parser.parse_args()
    print("argv:", " ".join(os.sys.argv), flush=True)

    if args.dir:
        root = os.path.abspath(args.dir)
        print(f"Batch mode: scanning {root}", flush=True)
        paths = list(iter_supported_files(root))
        print(f"Found {len(paths)} files:", flush=True)
        for fp in paths:
            print(" -", fp, flush=True)
        if not paths:
            print("No supported files found. Exiting.", flush=True)
            return
    else:
        fp = pick_file()
        if not fp:
            print("No file selected. Exiting.")
            return
        paths = [fp]

    for file_path in paths:
        try:
            print(f"\n=== Processing: {file_path}", flush=True)
            text = process_file(file_path)

            # if you added normalization/segmentation helpers, keep them; otherwise send raw text:
            segments = {"_all": text}

            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            base = Path(file_path).stem
            txt_out = OUTPUT_DIR / f"{base}.{ts}.txt"
            json_out = OUTPUT_DIR / f"{base}.{ts}.json"
            txt_out.write_text(text, encoding="utf-8")
            print(f"✔ Saved extracted text → {txt_out}", flush=True)

            # build messages (support both signatures)
            try:
                messages = build_messages(text, os.path.basename(file_path), segments)
            except TypeError:
                messages = build_messages(text, os.path.basename(file_path))

            print("→ Calling local LLM…", flush=True)
            raw = call_llm(messages)
            print(f"← LLM responded, {len(raw)} chars", flush=True)

            structured = parse_llm_json(raw)
            d = structured.model_dump() if hasattr(structured, "model_dump") else structured.dict()

            d["meta"] = {
                **d.get("meta", {}),
                "source_file": os.path.basename(file_path),
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "model": MODEL_NAME,
            }

            json_out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✔ Saved structured JSON → {json_out}", flush=True)

        except Exception as e:
            print(f"✖ Error on {file_path}: {e}", flush=True)

# ----------------------------
# CONFIG — adjust if needed
# ----------------------------
LLAMA_API_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "phi-3.5-mini-instruct-q5_k_s"  # doesn't have to exist server-side; friendly label
TEMPERATURE = 0.0
MAX_TOKENS = 3000  # enough for deep JSON
REQUEST_TIMEOUT = 90


# Tesseract path (you already set this)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Add Ghostscript (you already set this; keeping for safety)
gs_path = r"C:\Program Files\gs\gs10.05.1\bin"
if os.path.isdir(gs_path) and gs_path not in os.environ.get("PATH", ""):
    os.environ["PATH"] = gs_path + os.pathsep + os.environ.get("PATH", "")

OUTPUT_DIR = Path(__file__).parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ----------------------------
# OCR helpers
# ----------------------------
def preprocess_image_for_ocr(image: np.ndarray) -> np.ndarray:
    """Improve OCR on scanned pages."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    denoised_image = cv2.fastNlMeansDenoising(gray_image, None, 30, 7, 21)
    adaptive_thresh = cv2.adaptiveThreshold(
        denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    blurred_image = cv2.GaussianBlur(adaptive_thresh, (5, 5), 0)
    resized_image = cv2.resize(blurred_image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    return resized_image

def extract_text_from_pdf(pdf_path: str) -> str:
    out = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                out.append(page_text)
            else:
                # Fallback OCR
                im = page.to_image()
                open_cv_image = np.array(im.original)
                open_cv_image = open_cv_image[:, :, ::-1]  # RGB->BGR then preprocess expects RGB; convert back:
                open_cv_image = open_cv_image[:, :, ::-1]  # BGR->RGB
                processed = preprocess_image_for_ocr(open_cv_image)
                ocr_text = pytesseract.image_to_string(processed)
                if ocr_text.strip():
                    out.append(ocr_text)
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

    # tables
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
    return pytesseract.image_to_string(img).strip()

def process_image_file(image_path: str) -> str:
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pre = preprocess_image_for_ocr(img)
    return pytesseract.image_to_string(pre).strip()

# ----------------------------
# Quality heuristics
# ----------------------------
def is_low_quality_text(text: str, min_length=300, min_alpha_ratio=0.7, max_symbol_ratio=0.25, max_garbage_lines=0.35) -> bool:
    lines = text.strip().splitlines()
    total_lines = len(lines)
    if total_lines == 0:
        return True

    garbage_lines = 0
    total_alpha = 0
    total_chars = 0
    total_symbols = 0

    for line in lines:
        stripped = line.strip()
        total_chars += len(stripped)
        total_alpha += sum(c.isalpha() for c in stripped)
        total_symbols += sum(1 for c in stripped if not c.isalnum() and not c.isspace())
        if len(stripped) < 5 or re.fullmatch(r'[^a-zA-Z0-9]+', stripped or ""):
            garbage_lines += 1

    if total_chars == 0:
        return True

    alpha_ratio = total_alpha / max(total_chars, 1)
    symbol_ratio = total_symbols / max(total_chars, 1)
    garbage_line_ratio = garbage_lines / max(total_lines, 1)

    return (
        total_chars < min_length or
        alpha_ratio < min_alpha_ratio or
        symbol_ratio > max_symbol_ratio or
        garbage_line_ratio > max_garbage_lines
    )

import unicodedata
import phonenumbers
from rapidfuzz import fuzz, process as fuzzprocess
from langdetect import detect
from dateutil import parser as dateparser

def normalize_text_generic(text: str) -> str:
    # Normalize Unicode & whitespace, remove soft hyphens, collapse spaces
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    # fix excessive spaces and weird line wraps
    text = re.sub(r"[ \t]+", " ", text)
    # join lines that were clearly wrapped mid-sentence (ends with comma/word; next line lowercased)
    lines = [l.rstrip() for l in text.splitlines()]
    out = []
    for i, l in enumerate(lines):
        if out:
            prev = out[-1]
            if re.search(r"[a-zA-Z0-9,;]$", prev) and (l and l[:1].islower()):
                out[-1] = prev + " " + l.lstrip()
                continue
        out.append(l)
    text = "\n".join(out)
    # compress blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

# Heuristic heading detector: typography & shape, not vocabulary
def is_heading(line: str) -> bool:
    s = line.strip()
    if not s: return False
    # short-ish, title-like lines
    word_count = len(s.split())
    if word_count <= 8 and s.endswith(":"):
        return True
    # all-caps ratio
    letters = re.findall(r"[A-Za-z]", s)
    if letters:
        caps = sum(1 for ch in letters if ch.isupper())
        if caps / len(letters) >= 0.65 and word_count <= 8:
            return True
    # surrounded by blank lines often implies a heading
    return False

# Canonical buckets you care about (language-agnostic)
CANON_LABELS = [
    "profile", "experience", "education", "skills", "projects",
    "certifications", "languages", "awards", "publications",
    "volunteering", "references", "other"
]

# Optional minimal alias set (multilingual but tiny); safe to extend later
CANON_HINTS = {
    "experience": ["experience", "work", "employment", "career", "beruf", "erfahrung", "praxis"],
    "education": ["education", "academic", "studies", "ausbildung", "bildung", "studium", "school"],
    "skills": ["skills", "competencies", "kenntnisse", "fähigkeiten", "it", "computer"],
    "projects": ["projects", "projekt"],
    "certifications": ["certifications", "certificates", "zertifikat", "zertifikate"],
    "languages": ["languages", "sprachen", "idiomas", "langues"],
    "profile": ["profile", "summary", "objective", "über mich", "about"],
    "awards": ["awards", "honors", "preise"],
    "publications": ["publications", "publikationen"],
    "volunteering": ["volunteer", "ehrenamt", "community"],
    "references": ["references", "referees", "referenzen"],
}

def map_to_canonical(label: str) -> str:
    # Try fuzzy match on hints; fallback to 'other'
    label_l = label.lower().strip(" :")
    candidates = []
    for canon, hints in CANON_HINTS.items():
        # score on both the label itself and split tokens
        scores = [fuzz.partial_ratio(label_l, h) for h in hints + [canon]]
        candidates.append((max(scores), canon))
    best = max(candidates or [(0, "other")], key=lambda x: x[0])
    return best[1] if best[0] >= 70 else "other"

def segment_by_shape(text: str) -> dict:
    segments = {}
    current = "profile"  # start bucket
    segments[current] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if is_heading(line):
            # heading content (strip colon), map to canonical
            raw = re.sub(r":\s*$", "", line.strip())
            current = map_to_canonical(raw)
            segments.setdefault(current, [])
            continue
        segments[current].append(line)
    # finalize strings
    return {k: "\n".join(v).strip() for k, v in segments.items() if v and v[0].strip()}

# Deterministic contact extraction (language-agnostic)
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s\-\.])?(?:\(?\d{2,4}\)?[\s\-\.])?\d{3,5}[\s\-\.]?\d{3,5})")

def extract_contacts(text: str) -> dict:
    emails = sorted(set(EMAIL_RE.findall(text)))
    # try to parse phones via phonenumbers; if fail, fallback to regex strings
    raw_candidates = set(PHONE_RE.findall(text))
    phones = []
    for cand in raw_candidates:
        try:
            for m in phonenumbers.PhoneNumberMatcher(cand, "US"):  # region is just a default
                phones.append(phonenumbers.format_number(m.number, phonenumbers.PhoneNumberFormat.E164))
        except Exception:
            pass
    phones = sorted(set(phones)) or sorted(raw_candidates)
    return {"emails": emails, "phones": phones}

def detect_language(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"




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
        text = process_image_file(file_path)
        if not text:
            text = extract_text_from_image(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    if is_low_quality_text(text):
        print("⚠️  Warning: Extracted text looks low quality (scan or layout). Consider a clearer copy.")
    return text

# ----------------------------
# Schema (Pydantic) — many categories
# ----------------------------
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
    level: str = ""  # e.g., Native, C1, B2…

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
    contact: Dict[str, Any] = Field(default_factory=dict)  # emails, phones, address
    summary: str = ""
    keywords: List[str] = Field(default_factory=list)  # extracted key terms
    skills: Dict[str, List[str]] = Field(default_factory=dict)  # {hard:[], soft:[], tools:[], domains:[]}
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
    preferences: Dict[str, Any] = Field(default_factory=dict)  # relocation, remote, travel, salary, locations
    availability: str = ""
    meta: Dict[str, Any] = Field(default_factory=dict)  # parsing notes, confidence, source filename

# ----------------------------
# Prompt for strict JSON extraction
# ----------------------------
JSON_INSTRUCTIONS = """
You are a structured data extractor. Return ONLY valid JSON that conforms to the given schema.
- No commentary, no markdown, no code fences.
- Fill as many fields as possible from the text.
- Normalize dates to ISO-like strings when possible (YYYY-MM or YYYY).
- Derive categories aggressively (hard vs soft skills, tools, domains).
- Extract keywords (10–40) that best represent the profile.
- If a field is unknown, keep it as an empty string or empty array.
"""

COMPACT_SCHEMA = {
  "full_name": "", "job_title": "", "contact": {"emails": [], "phones": [], "address": ""},
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


def build_messages(resume_text: str, filename: str, segments: dict = None):
    seg_dump = json.dumps(segments or {"_all": resume_text}, ensure_ascii=False)
    return [
        {"role": "system",
         "content": "Return ONLY valid JSON. Output must start with '{' and end with '}'. "
                    "Unknown fields = empty strings/arrays. No markdown or commentary."},
        {"role": "user",
         "content": (
             JSON_INSTRUCTIONS.strip()
             + "\n\nSCHEMA (compact):\n" + json.dumps(COMPACT_SCHEMA, ensure_ascii=False)
             + "\n\nTEXT (or SEGMENTS):\n" + seg_dump
             + "\n\nFILENAME: " + filename
         )}
    ]



# ----------------------------
# LLM call (local server)
# ----------------------------
# class LLMError(Exception):
    pass

# @retry(
#     reraise=True,
#     stop=stop_after_attempt(3),
#     wait=wait_exponential(multiplier=1, min=2, max=10),
#     retry=retry_if_exception_type(LLMError)
# )
REQUEST_CONNECT_TIMEOUT = 5
REQUEST_READ_TIMEOUT = 300  # 5 min read window

def call_llm(messages):
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
        # keep this ONLY if your server supports it; otherwise remove it:
        # "response_format": {"type": "json_object"},
    }
    headers = {"Content-Type": "application/json", "Authorization": "Bearer local"}
    try:
        print(f"[LLM] POST {LLAMA_API_URL}  (connect {REQUEST_CONNECT_TIMEOUT}s, read {REQUEST_READ_TIMEOUT}s)", flush=True)
        resp = requests.post(
            LLAMA_API_URL,
            json=payload,
            headers=headers,
            timeout=(5, 300),
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
    # Remove trailing commas before a closing } or ]
    return re.sub(r',\s*([}\]])', r'\1', s)

def _auto_close_brackets(s: str) -> str:
    stack = []
    pairs = {'{': '}', '[': ']'}
    opens = set(pairs.keys())
    closes = set(pairs.values())

    out = []
    for ch in s:
        out.append(ch)
        if ch in opens:
            stack.append(ch)
        elif ch in closes:
            if stack and pairs.get(stack[-1]) == ch:
                stack.pop()
            else:
                # mismatch; drop it
                out.pop()

    # Close whatever is still open, in reverse
    while stack:
        out.append(pairs[stack.pop()])
    return ''.join(out)

def parse_llm_json(raw: str) -> ResumeJSON:
    # Keep only the first {...} block if the model added extra text
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    candidate = raw if (first_brace == -1 or last_brace == -1) else raw[first_brace:last_brace+1]

    # First attempt: as-is
    try:
        obj = json.loads(candidate)
        return ResumeJSON(**obj)
    except Exception:
        pass

    # Repair attempt: strip trailing commas and auto-close brackets
    repaired = _auto_close_brackets(_strip_trailing_commas(candidate))

    # If we still fail, also try to trim to the last complete JSON token boundary
    try:
        obj = json.loads(repaired)
    except json.JSONDecodeError:
        # Try to progressively trim the tail and parse
        for cut in range(len(repaired), max(len(repaired) - 2000, 0), -50):
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
        # If schema validation fails, still return what we can—but raise a clear error
        raise LLMError(f"JSON did not match schema: {ve}\nJSON:\n{json.dumps(obj, indent=2)[:1500]}")

# ----------------------------
# UI: file picker & main flow
# ----------------------------
def pick_file() -> str:
    root = Tk()
    root.withdraw()
    root.update()
    filetypes = [
        ("All supported", "*.pdf *.docx *.odt *.png *.jpg *.jpeg"),
        ("PDF", "*.pdf"),
        ("Word DOCX", "*.docx"),
        ("OpenDocument", "*.odt"),
        ("Images", "*.png *.jpg *.jpeg"),
    ]
    path = filedialog.askopenfilename(title="Select a resume file", filetypes=filetypes)
    root.destroy()
    return path

def chunk_text_by_chars(text: str, max_chars: int = 6000, overlap: int = 500):
    """Simple char-based chunking with overlap; avoids blowing context."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        # try to end on a line break
        nl = text.rfind("\n", start, end)
        if nl != -1 and nl > start + 1000:
            end = nl
        chunks.append(text[start:end].strip())
        start = max(end - overlap, end)  # next chunk with small overlap
    return [c for c in chunks if c]

def merge_structured_json(a: dict, b: dict) -> dict:
    """Merge two ResumeJSON-shaped dicts."""
    out = json.loads(json.dumps(a))  # deep copy

    # simple fields: prefer longer/non-empty
    for k in ["full_name", "job_title", "summary", "availability"]:
        va, vb = out.get(k, ""), b.get(k, "")
        out[k] = vb if (vb and len(vb) > len(va)) else va

    # contact
    out.setdefault("contact", {})
    for key in ["email", "emails", "phones", "address"]:
        va = out["contact"].get(key, [] if key.endswith("s") else "")
        vb = b.get("contact", {}).get(key, [] if key.endswith("s") else "")
        if isinstance(va, list) or isinstance(vb, list):
            sa = set(va if isinstance(va, list) else ([va] if va else []))
            sb = set(vb if isinstance(vb, list) else ([vb] if vb else []))
            out["contact"][key] = sorted(sa | sb)
        else:
            out["contact"][key] = vb or va

    # list sections: concat + de-dup (by tuple of fields)
    def merge_list(section, key_tuple):
        out.setdefault(section, [])
        seen = set()
        for rec in out[section] + b.get(section, []):
            t = tuple(rec.get(k, "").strip().lower() if isinstance(rec.get(k), str) else str(rec.get(k)) for k in key_tuple)
            if t not in seen:
                seen.add(t)
                out[section].append(rec)

    merge_list("experience", ("company","title","start","end","location"))
    merge_list("education", ("institution","degree","start","end","location"))
    merge_list("projects", ("name","role","start","end"))
    merge_list("certifications", ("name","issuer","date"))
    merge_list("publications", ("title","venue","date"))
    merge_list("languages", ("language","level"))

    # simple lists: union
    for k in ["volunteering","awards","clearances","keywords","links"]:
        va = out.get(k, [])
        vb = b.get(k, [])
        if isinstance(va, list) and isinstance(vb, list):
            # For 'links' (list of dicts), do set on (platform,url)
            if k == "links":
                seen = set()
                merged = []
                for rec in va + vb:
                    t = (rec.get("platform","").lower(), rec.get("url","").lower())
                    if t not in seen:
                        seen.add(t)
                        merged.append(rec)
                out[k] = merged
            else:
                out[k] = sorted(set([json.dumps(x, sort_keys=True) for x in (va+vb)]))
                out[k] = [json.loads(x) if x.startswith("{") or x.startswith("[") else x for x in out[k]]
        else:
            out[k] = vb or va

    # skills: per-bucket union
    out.setdefault("skills", {})
    for bucket in ["hard","soft","tools","domains"]:
        sa = set(out["skills"].get(bucket, []))
        sb = set(b.get("skills", {}).get(bucket, []))
        merged = sorted(sa | sb)
        if merged:
            out["skills"][bucket] = merged

    # preferences/meta: shallow update, prefer b when set
    out.setdefault("preferences", {})
    out["preferences"] = {**out["preferences"], **b.get("preferences", {})}
    out.setdefault("meta", {})
    out["meta"] = {**out["meta"], **b.get("meta", {})}
    return out

def extract_structured_in_chunks(full_text: str, filename: str, build_messages_fn) -> dict:
    chunks = chunk_text_by_chars(full_text, max_chars=6000, overlap=400)
    merged = None
    for i, ch in enumerate(chunks, 1):
        seg = {"chunk": i, "text": ch}
        try:
            msgs = build_messages_fn(ch, filename, {"_chunk": ch})
        except TypeError:
            msgs = build_messages_fn(ch, filename)
        print(f"[CHUNK {i}/{len(chunks)}] → LLM", flush=True)
        raw = call_llm(msgs)
        part = parse_llm_json(raw)
        d = part.model_dump() if hasattr(part, "model_dump") else part.dict()
        if merged is None:
            merged = d
        else:
            merged = merge_structured_json(merged, d)
    return merged or {}


def chunk_text_by_chars(text: str, max_chars: int = 6000, overlap: int = 400):
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
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
            out["contact"][key] = vb or va or ( [] if key.endswith("s") else "" )

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
        if isinstance(va, list) and isinstance(vb, list):
            out[k] = sorted(set(va) | set(vb))
        else:
            out[k] = vb or va or []

    out.setdefault("skills", {})
    for bucket in ["hard","soft","tools","domains"]:
        sa = set(out["skills"].get(bucket, []))
        sb = set(b.get("skills", {}).get(bucket, []))
        merged = sorted(sa | sb)
        if merged: out["skills"][bucket] = merged

    out["preferences"] = {**out.get("preferences", {}), **b.get("preferences", {})}
    out["meta"] = {**out.get("meta", {}), **b.get("meta", {})}
    return out

def extract_structured_in_chunks(full_text: str, filename: str, build_messages_fn):
    chunks = chunk_text_by_chars(full_text, max_chars=6000, overlap=400)
    merged = None
    for i, ch in enumerate(chunks, 1):
        print(f"[CHUNK {i}/{len(chunks)}] → LLM", flush=True)
        try:
            msgs = build_messages_fn(ch, filename, {"_chunk": ch})
        except TypeError:
            msgs = build_messages_fn(ch, filename)
        raw = call_llm(msgs)
        part = parse_llm_json(raw)
        d = part.model_dump() if hasattr(part, "model_dump") else part.dict()
        merged = d if merged is None else merge_structured_json(merged, d)
    return merged or {}




def main():
    print("=== Rovari Local Resume Structurer ===")
    file_path = pick_file()
    if not file_path:
        print("No file selected. Exiting.")
        return

    if not os.path.exists(file_path):
        print("File does not exist.")
        return

    print(f"Selected: {file_path}")
    text = process_file(file_path)
    text = normalize_text_generic(text)
    segments = segment_by_shape(text)
    contacts = extract_contacts(text)  # keep for post-merge
    lang = detect_language(text)


    # Save raw text
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(file_path).stem
    txt_out = OUTPUT_DIR / f"{base}.{ts}.txt"
    json_out = OUTPUT_DIR / f"{base}.{ts}.json"

    txt_out.write_text(text, encoding="utf-8")
    print(f"✔ Saved extracted text → {txt_out}")

    # Build prompt and call local LLM
    messages = build_messages(text, os.path.basename(file_path), segments)
    print("Calling local LLM for JSON structuring…")
    raw = call_llm(messages)

    # Parse & validate
    try:
        structured = parse_llm_json(raw)
    except LLMError as e:
        # Save the raw for debugging
        (OUTPUT_DIR / f"{base}.{ts}.raw.json.txt").write_text(raw, encoding="utf-8")
        print("✖ LLM JSON error.")
        print(str(e))
        print("Raw LLM output saved for inspection.")
        return

    # Attach meta and save
    d = structured.model_dump()
    # Merge deterministic contacts if LLM missed them
    d.setdefault("contact", {})
    if contacts.get("emails") and not (d["contact"].get("email") or d["contact"].get("emails")):
        d["contact"]["emails"] = contacts["emails"]
    if contacts.get("phones") and not (d["contact"].get("phones")):
        d["contact"]["phones"] = contacts["phones"]

    # add lightweight meta
    d["meta"] = {
        **d.get("meta", {}),
        "language": lang,
        "source_file": os.path.basename(file_path),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    main()
