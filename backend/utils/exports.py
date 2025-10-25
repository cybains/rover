import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict


def _base_dir(session_id: str, export_dir: str) -> Path:
    day = datetime.utcnow().strftime("%Y-%m-%d")
    path = Path(export_dir) / day / session_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_txts(session_id: str, segments: List[Dict], export_dir: str):
    base = _base_dir(session_id, export_dir)
    src = base / "transcript_src.txt"
    en = base / "translation_en.txt"
    with src.open("w", encoding="utf-8") as fs, en.open("w", encoding="utf-8") as fe:
        for s in segments:
            fs.write(s.get("textSrc", "") + "\n")
            fe.write(s.get("textEn", "") + "\n")


def write_jsonl(session_id: str, segments: List[Dict], export_dir: str):
    base = _base_dir(session_id, export_dir)
    path = base / "segments.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for s in segments:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
