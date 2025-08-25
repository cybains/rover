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



def run_one_with_classification(file_path: str, force_type: str = None):
    print(f"\n=== Processing: {file_path}", flush=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(file_path).stem
    txt_out = rs.OUTPUT_DIR / f"{base}.{ts}.txt"
    json_out = rs.OUTPUT_DIR / f"{base}.{ts}.json"
    err_out = rs.OUTPUT_DIR / f"{base}.{ts}.error.txt"

    try:
        text = rs.process_file(file_path)
        try:
            text = rs.normalize_text_generic(text)
            segments = rs.segment_by_shape(text)
            contacts = rs.extract_contacts(text)
            lang = rs.detect_language(text)
        except Exception:
            segments = {"_all": text}; contacts = {}; lang = "unknown"

        txt_out.write_text(text, encoding="utf-8")
        print(f"✔ Saved extracted text → {txt_out}", flush=True)

        from resume_structurer import classify_document, extract_id_document_fields, TRY_LLM_ON_NONRESUME
        doc_type = force_type or classify_document(text, os.path.basename(file_path))
        print(f"• Classified as '{doc_type}'", flush=True)

        # Base JSON (always defined)
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

        if doc_type == "resume":
            try:
                if hasattr(rs, "extract_structured_in_chunks"):
                    print("[CHUNK 1/…] → LLM (chunked extraction)", flush=True)
                    d2 = rs.extract_structured_in_chunks(text, os.path.basename(file_path), rs.build_messages)
                else:
                    print("→ Calling local LLM…", flush=True)
                    raw = rs.call_llm(rs.build_messages(text, os.path.basename(file_path), segments))
                    structured = rs.parse_llm_json(raw)
                    d2 = structured.model_dump() if hasattr(structured, "model_dump") else structured.dict()
                try:
                    d2 = rs.postprocess_structured(d2, contacts, lang)
                except Exception:
                    pass
                d = rs.merge_structured_json(d, d2)
            except Exception as e:
                msg = f"Resume parsing failed, wrote minimal JSON. Error: {e}"
                print(f"ℹ️  {msg}", flush=True)
                err_out.write_text(msg, encoding="utf-8")
        else:
            try:
                id_doc = extract_id_document_fields(text)
                if id_doc:
                    d["meta"]["id_document"] = id_doc
            except Exception as e:
                print(f"ℹ️  ID extraction skipped: {e}", flush=True)

            if TRY_LLM_ON_NONRESUME:
                try:
                    msgs = rs.build_messages(text, os.path.basename(file_path), {"_all": text, "_doc_type": doc_type})
                    raw = rs.call_llm(msgs)
                    part = rs.parse_llm_json(raw)
                    part_d = part.model_dump() if hasattr(part, "model_dump") else part.dict()
                    d = rs.merge_structured_json(d, part_d)
                except Exception as e:
                    msg = f"LLM on non-resume failed softly: {e}"
                    print(f"ℹ️  {msg}", flush=True)
                    err_out.write_text(msg, encoding="utf-8")
            else:
                print(f"ℹ️  Non-resume '{doc_type}'. Skipped LLM.", flush=True)

        d.setdefault("meta", {})
        d["meta"].update({
            "source_file": os.path.basename(file_path),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "model": rs.MODEL_NAME,
            "doc_type": d["meta"].get("doc_type", doc_type),
        })
        rs.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"✔ Saved structured JSON → {json_out}", flush=True)

    except Exception as e:
        err_text = f"Fatal error on {file_path}: {e}"
        print(f"✖ {err_text}", flush=True)
        try: err_out.write_text(err_text, encoding="utf-8")
        except Exception: pass
        minimal = {
            "full_name": "", "job_title": "", "contact": {"emails": [], "phones": [], "address": ""},
            "summary": "", "keywords": [], "skills": {"hard": [], "soft": [], "tools": [], "domains": []},
            "experience": [], "education": [], "projects": [], "certifications": [], "languages": [],
            "publications": [], "volunteering": [], "awards": [], "links": [], "clearances": [],
            "preferences": {}, "availability": "",
            "meta": {
                "error": err_text, "source_file": os.path.basename(file_path),
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "model": rs.MODEL_NAME, "doc_type": "other"
            }
        }
        try:
            json_out.write_text(json.dumps(minimal, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✔ Wrote minimal JSON with error → {json_out}", flush=True)
        except Exception:
            print("✖ Could not write minimal JSON file.", flush=True)


if __name__ == "__main__":
    main()
