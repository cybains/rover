# batch_runner.py — runs resume_structurer in batch (no GUI)
import os, json, sys
from pathlib import Path
from datetime import datetime
import resume_structurer as rs  # this file must be next to resume_structurer.py
from resume_structurer import classify_document  # add this

def pf(msg): print(msg, flush=True)

def iter_supported_files(root: str):
    exts = {".pdf", ".docx", ".odt", ".png", ".jpg", ".jpeg"}
    for p, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.join(p, f)

def run_one(file_path: str):
    pf(f"\n=== Processing: {file_path}")

    text = rs.process_file(file_path)
    norm = rs.normalize_text_generic(text)
    segments = rs.segment_by_shape(norm)
    contacts = rs.extract_contacts(norm)
    lang = rs.detect_language(norm)
        # 4) Classify document
    doc_type = classify_document(text, os.path.basename(file_path))
    print(f"• Classified as '{doc_type}'", flush=True)

    # 5) Get structured JSON dict `d`
    if doc_type == "resume":
        if hasattr(rs, "extract_structured_in_chunks"):
            print("[CHUNK 1/…] → LLM (chunked extraction)", flush=True)
            d = rs.extract_structured_in_chunks(text, os.path.basename(file_path), rs.build_messages)
        else:
            print("→ Calling local LLM…", flush=True)
            raw = rs.call_llm(messages)
            structured = rs.parse_llm_json(raw)
            d = structured.model_dump() if hasattr(structured, "model_dump") else structured.dict()

        # 6) Post-process
        try:
            d = rs.postprocess_structured(d, contacts, lang)
        except Exception:
            pass
    else:
        # Minimal JSON for non-resume docs
        d = {
            "full_name": "", "job_title": "", "contact": contacts, "summary": "",
            "keywords": [], "skills": {"hard": [], "soft": [], "tools": [], "domains": []},
            "experience": [], "education": [], "projects": [], "certifications": [],
            "languages": [], "publications": [], "volunteering": [], "awards": [],
            "links": [], "clearances": [], "preferences": {}, "availability": "",
            "meta": {
                "language": lang, "source_file": os.path.basename(file_path),
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "model": rs.MODEL_NAME, "doc_type": doc_type
            }
        }
        print(f"ℹ️  Non-resume '{doc_type}'. Skipped LLM.", flush=True)


    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(file_path).stem
    txt_out = rs.OUTPUT_DIR / f"{base}.{ts}.txt"
    json_out = rs.OUTPUT_DIR / f"{base}.{ts}.json"
    txt_out.write_text(norm, encoding="utf-8")
    pf(f"✔ Saved extracted text → {txt_out}")

    if doc_type != "resume":
        minimal = {
            "full_name": "", "job_title": "", "contact": contacts, "summary": "",
            "keywords": [], "skills": {"hard": [], "soft": [], "tools": [], "domains": []},
            "experience": [], "education": [], "projects": [], "certifications": [],
            "languages": [], "publications": [], "volunteering": [], "awards": [],
            "links": [], "clearances": [], "preferences": {}, "availability": "",
            "meta": {
                "source_file": os.path.basename(file_path),
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "model": rs.MODEL_NAME,
                "language": lang,
                "doc_type": doc_type
            }
        }
        json_out.write_text(json.dumps(minimal, indent=2, ensure_ascii=False), encoding="utf-8")
        pf(f"ℹ️  Classified as '{doc_type}'. Skipped LLM. → {json_out}")
        return

    # Resume path: chunked extraction + postprocess
    d = rs.extract_structured_in_chunks(norm, os.path.basename(file_path), rs.build_messages)
    d = rs.postprocess_structured(d, contacts, lang)
    d.setdefault("meta", {})
    d["meta"].update({
        "source_file": os.path.basename(file_path),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": rs.MODEL_NAME,
        "doc_type": "resume"
    })
    json_out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    pf(f"✔ Saved structured JSON → {json_out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        pf("Usage: python batch_runner.py <directory-or-file>")
        sys.exit(1)
    target = sys.argv[1]
    if os.path.isdir(target):
        files = list(iter_supported_files(target))
        pf(f"Batch mode: found {len(files)} files under {os.path.abspath(target)}")
        for fp in files:
            try:
                run_one(fp)
            except Exception as e:
                pf(f"✖ Error on {fp}: {e}")
    else:
        run_one(target)
