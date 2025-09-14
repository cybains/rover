#!/usr/bin/env python3
"""Preflight checks for development startup.

Verifies environment, model availability, MongoDB connection, and free ports.
"""
import os
import shutil
import socket
import sys

from pymongo import MongoClient
import torch


PORTS = [4001, 4002, 4000, 3000, 8080]


def check_env() -> None:
    device = os.getenv("ASR_DEVICE", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("ASR_DEVICE=cuda but CUDA not available")
    if shutil.which("llama-server") is None:
        print("warning: llama-server not found in PATH", file=sys.stderr)


def check_models() -> None:
    model_id = os.getenv("ASR_MODEL", "small")
    model_dir = os.path.expanduser(os.getenv("ASR_MODEL_DIR", "~/.cache"))
    if not os.path.isdir(model_dir):
        print(
            f"warning: model directory {model_dir} missing; Whisper will download {model_id}",
            file=sys.stderr,
        )


def check_mongo() -> None:
    url = os.getenv("MONGO_URL", "mongodb://localhost:27017")
    client = MongoClient(url, serverSelectionTimeoutMS=2000)
    client.server_info()


def check_ports() -> None:
    for port in PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                raise RuntimeError(f"port {port} is already in use")


def main() -> int:
    try:
        check_env()
        check_models()
        check_mongo()
        check_ports()
    except Exception as exc:  # pragma: no cover - best-effort script
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 1
    print("Preflight OK")
    return 0


if __name__ == "__main__":  # pragma: no cover - direct execution only
    raise SystemExit(main())
