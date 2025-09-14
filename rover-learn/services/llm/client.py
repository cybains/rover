from __future__ import annotations

import json
import os

import httpx

LLM_URL = os.getenv("LLM_URL", "http://127.0.0.1:8080")
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
