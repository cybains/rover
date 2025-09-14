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


def _recorder_with_blocksize(sr: int, frames_per_chunk: int, source: str):
    """
    Returns (recorder, is_stereo).
    Sets blocksize == frames_per_chunk to stabilize WASAPI buffering.
    """
    mode = (source or "auto").lower()

    if mode == "mic":
        mic = sc.default_microphone()
        if mic is None:
            print("[agent] microphone not found (check privacy settings).", file=sys.stderr)
            return None, False
        print("[agent] using microphone")
        return mic.recorder(samplerate=sr, channels=1, blocksize=frames_per_chunk), False

    if mode == "loopback":
        lb = _default_loopback_mic()
        if lb is None:
            print("[agent] loopback not found (no default speaker).", file=sys.stderr)
            return None, True
        print("[agent] using loopback")
        # some loopback devices insist on 2 channels
        return lb.recorder(samplerate=sr, channels=2, blocksize=frames_per_chunk), True

    # auto: prefer mic
    mic = sc.default_microphone()
    if mic is not None:
        print("[agent] using microphone (auto)")
        return mic.recorder(samplerate=sr, channels=1, blocksize=frames_per_chunk), False
    lb = _default_loopback_mic()
    if lb is not None:
        print("[agent] using loopback (auto)")
        return lb.recorder(samplerate=sr, channels=2, blocksize=frames_per_chunk), True

    print("[agent] no audio devices found", file=sys.stderr)
    return None, False


def _to_b64_pcm16(frames: np.ndarray, stereo: bool) -> str:
    """Convert float32 frames [-1, 1] to base64 PCM16 mono."""
    if frames.ndim == 2:
        frames = frames.mean(axis=1) if stereo else frames[:, 0]
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
    ap.add_argument("--chunk_ms", type=int, default=500, help="Chunk size in ms")
    args = ap.parse_args()

    session_id = args.session or os.environ.get("SESSION_ID")
    if not session_id:
        print("[agent] session id required via --session or SESSION_ID", file=sys.stderr)
        return 2

    if args.chunk_ms <= 0:
        print("[agent] chunk_ms must be > 0", file=sys.stderr)
        return 2

    sr = int(args.sr)
    frames_per_chunk = int(sr * (args.chunk_ms / 1000.0))

    rec, stereo = _recorder_with_blocksize(sr, frames_per_chunk, args.source)
    if rec is None:
        return 1

    print(
        f"[agent] device: {getattr(rec, 'microphone', None) and rec.microphone.name or 'unknown'}"
    )

    s = requests.Session()
    start_t = time.monotonic()
    consec_err = 0
    last_ok = time.time()
    hb_last = time.time()
    dropped = 0

    try:
        with rec:
            while True:
                # Capture ~chunk_ms of audio
                frames = rec.record(numframes=frames_per_chunk)  # float32 in [-1, 1]
                elapsed_ms = (time.monotonic() - start_t) * 1000.0
                idx = int(round(elapsed_ms / args.chunk_ms))

                if np.max(np.abs(frames)) < 1e-3:
                    idx = max(idx, 0)
                    continue

                b16 = _to_b64_pcm16(frames, stereo=stereo)

                # --- Send to ASR ---
                try:
                    t0 = time.time()
                    r = s.post(
                        f"{args.asr}/transcribe_chunk",
                        json={
                            "audio_b16": b16,
                            "sample_rate": sr,
                            "idx": idx,
                            "chunk_ms": args.chunk_ms,  # align server timeline
                        },
                        timeout=10,
                    )
                    r.raise_for_status()
                    payload = r.json()
                    segments = payload.get("segments", [])
                    asr_ms = (time.time() - t0) * 1000.0
                    if not segments:
                        dropped += 1
                    consec_err = 0
                except Exception as e:
                    print(f"[agent] ASR error: {e}", file=sys.stderr)
                    consec_err += 1
                    if consec_err >= 4:
                        time.sleep(0.2)
                    continue

                # --- Forward to backend ---
                for seg in segments:
                    seg["sessionId"] = session_id
                    seg["asrMs"] = asr_ms
                    try:
                        br = s.post(f"{args.api}/ingest_segment", json=seg, timeout=10)
                        br.raise_for_status()
                        last_ok = time.time()
                        consec_err = 0
                    except Exception as e:
                        print(f"[agent] backend ingest error: {e}", file=sys.stderr)
                        consec_err += 1
                        if consec_err >= 4:
                            time.sleep(0.2)

                # heartbeat
                if time.time() - hb_last > 5:
                    try:
                        s.post(
                            f"{args.api}/heartbeat",
                            json={"sessionId": session_id, "dropped": dropped},
                            timeout=5,
                        )
                    except Exception:
                        pass
                    dropped = 0
                    hb_last = time.time()

                # watchdog: show if nothing ingested for a while
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
