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
p = /*************  ✨ Windsurf Command ⭐  *************/
    """
    Capture audio from the default microphone or speaker and send it to the ASR service
    for transcription. The transcribed segments are forwarded to the backend.

    Parameters:
    --session (required): session ID
    --asr: ASR service URL (default: http://localhost:4001)
    --api: backend service URL (default: http://localhost:4000)
    --source: audio source (default: auto, choices: auto, mic, loopback)
    --sr: sample rate (default: 16000)
    """
/*******  644dccd2-b6e2-4273-a25d-03fafe42dfd3  *******/argparse.ArgumentParser()
    p.add_argument("--session")
    p.add_argument("--asr", default="http://localhost:4001")
    p.add_argument("--api", default="http://localhost:4000")
    p.add_argument("--source", choices=["auto", "mic", "loopback"], default="auto")
    args = p.parse_args()

    session_id = args.session or os.environ.get("SESSION_ID")
    if not session_id:
        print("session id required via --session or SESSION_ID")
        return

    sr = 16000
    rec, stereo = _recorder(sr, args.source)
    if rec is None:
        return 1
    idx = 0
    with rec as r:
        try:
            while True:
                data = r.record(numframes=sr)
                if stereo:
                    data = data.mean(axis=1)
                pcm16 = np.clip(data, -1, 1)
                pcm16 = (pcm16 * 32767).astype(np.int16)
                b16 = base64.b64encode(pcm16.tobytes()).decode("ascii")

                try:
                    res = requests.post(
                        f"{args.asr}/transcribe_chunk",
                        json={"audio_b16": b16, "sample_rate": sr, "idx": idx},
                        timeout=10,
                    ).json()
                except Exception:
                    res = {"segments": []}

                for seg in res.get("segments", []):
                    seg_payload = {**seg, "sessionId": session_id}
                    try:
                        requests.post(
                            f"{args.api}/ingest_segment", json=seg_payload, timeout=10
                        )
                    except Exception:
                        pass
                idx += 1
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

