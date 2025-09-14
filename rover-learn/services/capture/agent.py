"""Simple capture agent: Mic/Loopback -> ASR -> Backend."""

import argparse
import base64
import os

import numpy as np
import requests
import soundcard as sc


def _recorder(sr: int):
    mic = sc.default_microphone()
    if mic is not None:
        print("using microphone")
        return mic.recorder(samplerate=sr, channels=1), False
    spk = sc.default_speaker()
    print("using loopback")
    return spk.recorder(samplerate=sr, channels=2), True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--session")
    p.add_argument("--asr", default="http://localhost:4001")
    p.add_argument("--api", default="http://localhost:4000")
    args = p.parse_args()

    session_id = args.session or os.environ.get("SESSION_ID")
    if not session_id:
        print("session id required via --session or SESSION_ID")
        return

    sr = 16000
    rec, stereo = _recorder(sr)
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
    main()

