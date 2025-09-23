#!/usr/bin/env python3
"""
Capture agent:
- Starts an ASR session (mic or system)
- Subscribes to ASR SSE stream
- Sends partial and final segments to backend /ingest_segment
"""

import argparse
import json
import sys
import time
from typing import Optional

import httpx


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True, help="backend sessionId")
    ap.add_argument("--asr", default="http://localhost:5001", help="ASR base URL")
    ap.add_argument("--api", default="http://localhost:4000", help="Backend API base URL")
    ap.add_argument("--source", default="mic", choices=["mic", "system"], help="Audio source")
    return ap.parse_args()


def main():
    args = parse_args()
    session_id = args.session
    asr_base = args.asr.rstrip("/")
    api_base = args.api.rstrip("/")
    source = args.source

    client = httpx.Client(timeout=10.0)
    stream_client = httpx.Client(timeout=None)

    asr_id: Optional[str] = None

    try:
        # 1) Start ASR session, pass source (mic/system)
        r = client.post(f"{asr_base}/session/start", json={"source": source})
        r.raise_for_status()
        asr_id = r.json().get("session_id")

        # 2) Open SSE
        with stream_client.stream("GET", f"{asr_base}/session/{asr_id}/stream") as resp:
            resp.raise_for_status()

            last_partial_sent = 0.0

            for line in resp.iter_lines():
                if line is None:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="ignore")
                line = line.strip()
                if not line or not line.startswith("data:"):
                    continue

                payload = line[5:].strip()
                try:
                    ev = json.loads(payload)
                except Exception:
                    continue

                text = (ev.get("text") or "").strip()
                lang = (ev.get("lang") or "").lower()
                t_start = float(ev.get("start_sec") or 0.0)
                t_end = float(ev.get("end_sec") or max(t_start, t_start + 1.0))
                chunk_index = int(ev.get("chunk_index") or 0)
                is_final = bool(ev.get("final"))

                now = time.time()

                # 3a) Partial updates (throttle ~0.6s)
                if not is_final and text:
                    if now - last_partial_sent >= 0.6:
                        try:
                            client.post(
                                f"{api_base}/ingest_segment",
                                json={
                                    "sessionId": session_id,
                                    "kind": "partial",
                                    "lang": lang or "auto",
                                    "textSrcPartial": text,
                                    "speaker": "Speaker 1",
                                },
                                timeout=3.0,
                            )
                        except Exception:
                            pass
                        last_partial_sent = now

                # 3b) Final segments
                if is_final:
                    try:
                        client.post(
                            f"{api_base}/ingest_segment",
                            json={
                                "sessionId": session_id,
                                "kind": "final",
                                "idx": chunk_index,
                                "tStart": t_start,
                                "tEnd": t_end,
                                "lang": lang or "auto",
                                "textSrc": text,
                                "speaker": "Speaker 1",
                                "confidence": 0.9,
                            },
                            timeout=10.0,
                        )
                    except Exception:
                        pass

    finally:
        # 4) Stop ASR session for cleanup
        if asr_id:
            try:
                client.post(f"{asr_base}/session/stop", json={"session_id": asr_id})
            except Exception:
                pass
        try:
            client.close()
        except Exception:
            pass
        try:
            stream_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
