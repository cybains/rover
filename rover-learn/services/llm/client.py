from __future__ import annotations

import json
import os
import sys
import shutil
from pathlib import Path

import httpx

LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8080")


def _ensure_local_llama() -> None:
    """Ensure `llama-server` binary is discoverable.

    If the binary isn't available in PATH, also look for
    `llama/llama-server(.exe)` relative to repo root. This helps Windows
    setups where the executable lives in a `llama` folder.
    """

    # already on PATH
    if shutil.which("llama-server"):
        return

    root = Path(__file__).resolve().parents[2]
    exe_name = "llama-server.exe" if os.name == "nt" else "llama-server"
    local = root / "llama" / exe_name
    if local.exists():
        os.environ["PATH"] = str(local.parent) + os.pathsep + os.environ.get("PATH", "")
    else:
        print("warning: llama-server not found in PATH or local llama directory", file=sys.stderr)


_ensure_local_llama()

HTTP = httpx.AsyncClient(timeout=30.0)


async def generate_json(prompt: str) -> dict:
    r = await HTTP.post(
        f"{LLM_URL}/v1/chat/completions",
        json={
            "model": os.getenv("LLM_MODEL", "phi-3.5-mini"),
            "messages": [
                {"role": "system", "content": "Output JSON only."},
                {"role": "user", "content": prompt},
            ],
        },
    )
    r.raise_for_status()
    txt = r.json()["choices"][0]["message"]["content"]
    return json.loads(txt)
