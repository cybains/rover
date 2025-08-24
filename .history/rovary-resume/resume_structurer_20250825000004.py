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
MAX_TOKENS = 1500  # enough for deep JSON
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

def build_messages(resume_text: str, filename: str) -> List[Dict[str, str]]:
    schema = ResumeJSON.schema()
    return [
        {
            "role": "system",
            "content": "You are precise, concise, and return only valid JSON."
        },
        {
            "role": "user",
            "content": (
                JSON_INSTRUCTIONS.strip()
                + "\n\nSCHEMA:\n"
                + json.dumps(schema, indent=2)
                + "\n\nTEXT:\n<<<\n"
                + resume_text
                + "\n>>>\n\n"
                + f"FILENAME: {filename}\n"
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
        "stream": False
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

def parse_llm_json(raw: str) -> ResumeJSON:
    # Try to extract the first {...} block if the model added anything extraneous
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1:
        raw = raw[first_brace:last_brace+1]
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON from LLM: {e}\nRaw:\n{raw[:1000]}")
    try:
        return ResumeJSON(**obj)
    except ValidationError as ve:
        # If schema validation fails, keep the raw JSON but raise an informative error
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

    # Save raw text
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(file_path).stem
    txt_out = OUTPUT_DIR / f"{base}.{ts}.txt"
    json_out = OUTPUT_DIR / f"{base}.{ts}.json"

    txt_out.write_text(text, encoding="utf-8")
    print(f"✔ Saved extracted text → {txt_out}")

    # Build prompt and call local LLM
    messages = build_messages(text, os.path.basename(file_path))
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
    d = structured.dict()
    d["meta"] = {
        **d.get("meta", {}),
        "source_file": os.path.basename(file_path),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": MODEL_NAME,
    }
    json_out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✔ Saved structured JSON → {json_out}")
    print("Done.")

if __name__ == "__main__":
    main()
