"""Simple capture agent: Mic/Loopback -> ASR -> Backend."""

import argparse
import base64
import os

import numpy as np
import requests
import soundcard as sc

# --- add/replace these helpers near the top ---
import soundcard as sc


def _recorder(sr: int, source: str):
    """
    Returns (recorder, is_stereo)
    - mic:     default microphone
    - loopback: loopback 'microphone' created from default speaker
    - auto:    try mic, else loopback
    """
    source = (source or "auto").lower()

    if source == "mic":
        mic = sc.default_microphone()
        if mic is None:
            raise SystemExit("No default microphone available (Windows privacy?).")
        return mic.recorder(samplerate=sr, channels=1), False

    if source == "loopback":
        spk = sc.default_speaker()
        if spk is None:
            raise SystemExit("No default speaker available for loopback.")
        lb_mic = sc.get_microphone(spk.name, include_loopback=True)
        return lb_mic.recorder(samplerate=sr, channels=2), True

    # auto
    mic = sc.default_microphone()
    if mic is not None:
        return mic.recorder(samplerate=sr, channels=1), False
    spk = sc.default_speaker()
    if spk is None:
        raise SystemExit("No microphone or speaker found for auto source.")
    lb_mic = sc.get_microphone(spk.name, include_loopback=True)
    return lb_mic.recorder(samplerate=sr, channels=2), True


def _to_pcm16_base64(frames: np.ndarray, stereo: bool) -> str:
    """
    frames: float32 in [-1, 1], shape (numframes, channels)
    """
    if frames.ndim == 2:
        if stereo:
            # downmix stereo -> mono
            frames = frames.mean(axis=1)
        else:
            frames = frames[:, 0]
    # clamp and convert to int16 little-endian
    frames = np.clip(frames, -1.0, 1.0)
    pcm16 = (frames * 32767.0).astype("<i2")  # little-endian int16
    import base64
    return base64.b64encode(pcm16.tobytes()).decode("ascii")


def main():
    import argparse, requests, time
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--asr", default="http://localhost:4001")
    parser.add_argument("--api", default="http://localhost:4000")
    parser.add_argument("--source", default="auto", choices=["auto","mic","loopback"])
    parser.add_argument("--sr", type=int, default=16000)
    args = parser.parse_args()

    rec, stereo = _recorder(args.sr, args.source)
    print(f"using {args.source}")

    idx = 0
    with rec:
        while True:
            frames = rec.record(numframes=args.sr)  # ~1s chunk
            b64 = _to_pcm16_base64(frames.astype("float32"), stereo)

            # send to ASR
            r = requests.post(f"{args.asr}/transcribe_chunk", json={
                "audio_b16": b64, "sample_rate": args.sr, "idx": idx
            }, timeout=10)
            r.raise_for_status()
            data = r.json()

            # forward segments to backend
            for seg in data.get("segments", []):
                seg["sessionId"] = args.session
                requests.post(f"{args.api}/ingest_segment", json=seg, timeout=10)

            idx += 1



if __name__ == "__main__":
    raise SystemExit(main())

