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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("target", help="Directory or file path to process")
    parser.add_argument("--force-type", type=str, choices=["resume","certificate","passport","other"],
                        help="Override document classification for all files")
    args = parser.parse_args()

    target = args.target
    if os.path.isdir(target):
        files = list(iter_supported_files(target))
        pf(f"Batch mode: found {len(files)} files under {os.path.abspath(target)}")
        for fp in files:
            try:
                run_one_with_classification(fp, force_type=args.force_type)
            except Exception as e:
                pf(f"✖ Error on {fp}: {e}")
    else:
        run_one_with_classification(target, force_type=args.force_type)


def run_one_with_classification(file_path: str, force_type: str = None):
    print(f"\n=== Processing: {file_path}", flush=True)

    # 1) Extract raw text
    text = rs.process_file(file_path)

    # 2) Normalize/segment + contacts/lang (best-effort)
    try:
        text = rs.normalize_text_generic(text)
        segments = rs.segment_by_shape(text)
        contacts = rs.extract_contacts(text)
        lang = rs.detect_language(text)
    except Exception:
        segments = {"_all": text}
        contacts, lang = {}, "unknown"

    # 3) Save extracted text
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(file_path).stem
    txt_out = rs.OUTPUT_DIR / f"{base}.{ts}.txt"
    json_out = rs.OUTPUT_DIR / f"{base}.{ts}.json"
    txt_out.write_text(text, encoding="utf-8")
    print(f"✔ Saved extracted text → {txt_out}", flush=True)

    # 4) Classify
    try:
        from resume_structurer import classify_document, extract_id_document_fields, TRY_LLM_ON_NONRESUME
    except Exception:
        # Fallback imports if names differ
        from resume_structurer import classify_document
        TRY_LLM_ON_NONRESUME = True
        def extract_id_document_fields(_t): return {}

    doc_type = force_type or classify_document(text, os.path.basename(file_path))
    print(f"• Classified as '{doc_type}'", flush=True)

    # 5) Build messages (fallback if build_messages doesn't accept segments)
    try:
        messages = rs.build_messages(text, os.path.basename(file_path), segments)
    except TypeError:
        messages = rs.build_messages(text, os.path.basename(file_path))

    # 6) Resume vs non-resume
    if doc_type == "resume":
        if hasattr(rs, "extract_structured_in_chunks"):
            print("[CHUNK 1/…] → LLM (chunked extraction)", flush=True)
            d = rs.extract_structured_in_chunks(text, os.path.basename(file_path), rs.build_messages)
        else:
            print("→ Calling local LLM…", flush=True)
            raw = rs.call_llm(messages)
            structured = rs.parse_llm_json(raw)
            d = structured.model_dump() if hasattr(structured, "model_dump") else structured.dict()

        # Post-process
        try:
            d = rs.postprocess_structured(d, contacts, lang)
        except Exception:
            pass
    else:
        # Non-resume scaffold + deterministic ID extraction + optional LLM merge
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
                "language": lang, "source_file": os.path.basename(file_path),
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "model": rs.MODEL_NAME, "doc_type": doc_type
            }
        }
        try:
            id_doc = extract_id_document_fields(text)
            if id_doc:
                d["meta"]["id_document"] = id_doc
        except Exception as _e:
            print(f"ℹ️  ID extraction skipped: {_e}", flush=True)

        if TRY_LLM_ON_NONRESUME:
            try:
                msgs = rs.build_messages(text, os.path.basename(file_path), {"_all": text, "_doc_type": doc_type})
                raw = rs.call_llm(msgs)
                part = rs.parse_llm_json(raw)
                part_d = part.model_dump() if hasattr(part, "model_dump") else part.dict()
                d = rs.merge_structured_json(d, part_d)
            except Exception as _e:
                print(f"ℹ️  LLM on non-resume failed softly: {_e}", flush=True)
        else:
            print(f"ℹ️  Non-resume '{doc_type}'. Skipped LLM.", flush=True)

    # 7) Meta + save
    d.setdefault("meta", {})
    d["meta"].update({
        "source_file": os.path.basename(file_path),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": rs.MODEL_NAME,
        "doc_type": doc_type,
    })
    rs.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✔ Saved structured JSON → {json_out}", flush=True)


if __name__ == "__main__":
    main()
