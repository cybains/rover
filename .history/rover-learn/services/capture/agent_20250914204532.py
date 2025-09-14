"""Simple capture agent: Mic/Loopback -> ASR -> Backend (1s chunks, 16 kHz, PCM16)."""

from __future__ import annotations

import argparse
import base64
import os
import sys
import time
import warnings

import numpy as np
import requests
import soundcard as sc

# silence noisy soundcard warnings
warnings.simplefilter("ignore", category=UserWarning)


def _default_loopback_mic() -> sc.Microphone | None:
    """Return a loopback microphone from the default speaker (Windows system audio)."""
    spk = sc.default_speaker()
    if spk is None:
        return None
    return sc.get_microphone(spk.name, include_loopback=True)


def _recorder(sr: int, source: str):
    """
    Returns (recorder, is_stereo)
    - mic: default microphone, mono
    - loopback: loopback mic from default speaker, stereo (downmixed)
    - auto: try mic first, fallback to loopback
    """
    mode = (source or "auto").lower()

    if mode == "mic":
        mic = sc.default_microphone()
        if mic is None:
            print("[agent] microphone not found (check privacy settings).", file=sys.stderr)
            return None, False
        print("[agent] using microphone")
        return mic.recorder(samplerate=sr, channels=1), False

    if mode == "loopback":
        lb = _default_loopback_mic()
        if lb is None:
            print("[agent] loopback not found (no default speaker).", file=sys.stderr)
            return None, True
        print("[agent] using loopback")
        return lb.recorder(samplerate=sr, channels=2), True

    # auto
    mic = sc.default_microphone()
    if mic is not None:
        print("[agent] using microphone (auto)")
        return mic.recorder(samplerate=sr, channels=1), False

    lb = _default_loopback_mic()
    if lb is not None:
        print("[agent] using loopback (auto)")
        return lb.recorder(samplerate=sr, channels=2), True

    print("[agent] no audio devices found", file=sys.stderr)
    return None, False


def _to_b64_pcm16(frames: np.ndarray, stereo: bool) -> str:
    """Convert float32 frames [-1, 1] to base64 PCM16 mono."""
    if frames.ndim == 2:
        if stereo:
            frames = frames.mean(axis=1)  # downmix
        else:
            frames = frames[:, 0]

    frames = np.asarray(frames, dtype=np.float32)
    np.clip(frames, -1.0, 1.0, out=frames)
    pcm16 = (frames * 32767.0).astype("<i2")  # int16 little-endian
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


def main() -> int:
    ap = argparse.ArgumentParser(description="Mic/Loopback -> ASR (/transcribe_chunk) -> Backend (/ingest_segment)")
    ap.add_argument("--session", help="Session ID (required, or set SESSION_ID env)")
    ap.add_argument("--asr", default="http://localhost:4001", help="ASR base URL")
    ap.add_argument("--api", default="http://localhost:4000", help="Backend base URL")
    ap.add_argument("--source", choices=["auto", "mic", "loopback"], default="auto", help="Audio source")
    ap.add_argument("--sr", type=int, default=16000, help="Sample rate (Hz)")
    ap.add_argument("--chunk_ms", type=int, default=1000, help="Chunk size in ms")
    args = ap.parse_args()

    session_id = args.session or os.environ.get("SESSION_ID")
    if not session_id:
        print("[agent] session id required via --session or SESSION_ID", file=sys.stderr)
        return 2

    sr = int(args.sr)
    frames_per_chunk = int(sr * (args.chunk_ms / 1000.0))

    rec, stereo = _recorder(sr, args.source)
    if rec is None:
        return 1

    s = requests.Session()
    idx = 0
    last_ok = time.time()

    try:
        with rec:
            while True:
                frames = rec.record(numframes=frames_per_chunk)  # float32 in [-1, 1]
                b16 = _to_b64_pcm16(frames, stereo=stereo)

                # --- Send to ASR ---
                try:
                    r = s.post(
                        f"{args.asr}/transcribe_chunk",
                        json={"audio_b16": b16, "sample_rate": sr, "idx": idx},
                        timeout=10,
                    )
                    r.raise_for_status()
                    payload = r.json()
                    segments = payload.get("segments", [])
                except Exception as e:
                    print(f"[agent] ASR error: {e}", file=sys.stderr)
                    segments = []

                # --- Forward to backend ---
                for seg in segments:
                    seg["sessionId"] = session_id
                    try:
                        br = s.post(f"{args.api}/ingest_segment", json=seg, timeout=10)
                        br.raise_for_status()
                        last_ok = time.time()
                    except Exception as e:
                        print(f"[agent] backend ingest error: {e}", file=sys.stderr)

                idx += 1

                # watchdog
                if time.time() - last_ok > 10:
                    print("[agent] no successful ingest for >10s", file=sys.stderr)
                    last_ok = time.time()

    except KeyboardInterrupt:
        print("\n[agent] stopped by user")
    except Exception as e:
        print(f"[agent] fatal error: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
