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

# ----------------------------
# CONFIG — adjust if needed
# ----------------------------
LLAMA_API_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "phi-3.5-mini-instruct-q5_k_s"  # doesn't have to exist server-side; friendly label
TEMPERATURE = 0.1
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

def build_messages(resume_text: str, filename: str, segments: dict) -> List[Dict[str, str]]:
    schema = ResumeJSON.model_json_schema()
    seg_dump = json.dumps(segments, ensure_ascii=False, indent=2)
    return [
        {
            "role": "system",
            "content": (
                "You are a structured data extractor. Return ONLY valid JSON. "
                "Output must start with '{' and end with '}'. "
                "If info is unknown, use empty strings/arrays. No text outside JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                JSON_INSTRUCTIONS.strip()
                + "\n\nSCHEMA:\n" + json.dumps(schema, indent=2)
                + "\n\nSEGMENTS:\n" + seg_dump
                + "\n\nFILENAME: " + filename
            ),
        },
    ]


# ----------------------------
# LLM call (local server)
# ----------------------------
class LLMError(Exception):
    pass

@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(LLMError)
)
def call_llm(messages: List[Dict[str, str]]) -> str:
    payload = {
    "model": MODEL_NAME,
    "messages": messages,
    "temperature": TEMPERATURE,
    "max_tokens": MAX_TOKENS,
    "stream": False,
    # Try strict JSON if supported by your llama-server build:
    "response_format": {"type": "json_object"}
    }

    try:
        resp = requests.post(LLAMA_API_URL, json=payload, timeout=REQUEST_TIMEOUT)
    except Exception as e:
        raise LLMError(f"Connection error to local LLM: {e}")

    if resp.status_code != 200:
        raise LLMError(f"LLM HTTP {resp.status_code}: {resp.text[:500]}")

    data = resp.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except Exception:
        raise LLMError(f"Unexpected LLM response: {data}")

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
