# batch_runner.py — runs resume_structurer in batch (no GUI)
import os, json, time, sys
from pathlib import Path
from datetime import datetime

import resume_structurer as rs  # your existing file

def iter_supported_files(root: str):
    exts = {".pdf", ".docx", ".odt", ".png", ".jpg", ".jpeg"}
    for p, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                yield os.path.join(p, f)

def pf(msg): print(msg, flush=True)

def run_one(file_path: str):
    pf(f"\n=== Processing: {file_path}")
    text = rs.process_file(file_path)
    

    # if you added normalizers/segmenters, use them; otherwise keep raw text
    segments = {"_all": text}

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    base = Path(file_path).stem
    txt_out = rs.OUTPUT_DIR / f"{base}.{ts}.txt"
    json_out = rs.OUTPUT_DIR / f"{base}.{ts}.json"
    txt_out.write_text(text, encoding="utf-8")
    pf(f"✔ Saved extracted text → {txt_out}")

    # build messages (support both signatures)
    try:
        messages = rs.build_messages(text, os.path.basename(file_path), segments)
    except TypeError:
        messages = rs.build_messages(text, os.path.basename(file_path))

    pf("→ Calling local LLM…")
    t0 = time.time()
    
    d = rs.extract_structured_in_chunks(text, os.path.basename(file_path), rs.build_messages)
    d["meta"] = {
        **d.get("meta", {}),
        "source_file": os.path.basename(file_path),
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "model": rs.MODEL_NAME,
    }
    json_out.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    pf(f"✔ Saved structured JSON → {json_out}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python batch_runner.py <directory-or-file>", flush=True)
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
