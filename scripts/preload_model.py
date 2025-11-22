#!/usr/bin/env python3
"""
Download and cache a Whisper model using faster-whisper.

Usage:
  python scripts/preload_model.py --model medium --compute-type float16
"""

from __future__ import annotations

import argparse
from pathlib import Path

from faster_whisper import WhisperModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preload Whisper model into local cache.")
    parser.add_argument("--model", default="medium", help="Whisper model size (e.g., tiny, base, small, medium, large)")
    parser.add_argument(
        "--compute-type",
        default="float16",
        help="faster-whisper compute type (e.g., float16, int8, int8_float16)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Optional directory to store the downloaded model (defaults to huggingface cache).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Target device for inference (auto, cpu, cuda, metal).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Downloading model '{args.model}' with compute_type '{args.compute_type}' on device '{args.device}'...")
    try:
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=args.compute_type,
            download_root=str(args.cache_dir) if args.cache_dir else None,
        )
        _ = model  # ensure weights are loaded
        print("Model cached.")
    except ValueError as exc:
        print(f"Error: {exc}")
        fallback = "int8" if args.compute_type != "int8" else "float32"
        print("Retrying with compute_type='%s'..." % fallback)
        model = WhisperModel(
            args.model,
            device=args.device,
            compute_type=fallback,
            download_root=str(args.cache_dir) if args.cache_dir else None,
        )
        _ = model
        print(f"Model cached using compute_type='{fallback}'.")


if __name__ == "__main__":
    main()
