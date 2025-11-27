from __future__ import annotations

import logging
from typing import Iterable

import numpy as np

logger = logging.getLogger(__name__)

try:
    import webrtcvad
except Exception:  # pragma: no cover - optional dependency
    webrtcvad = None  # type: ignore


class VADUnavailable(RuntimeError):
    pass


def _to_int16_pcm(audio: np.ndarray) -> np.ndarray:
    """Convert float32 [-1,1] audio to int16 PCM."""
    if audio.dtype != np.int16:
        clipped = np.clip(audio, -1.0, 1.0)
        audio = (clipped * 32767).astype(np.int16)
    return audio


def _frame_generator(audio: np.ndarray, sample_rate: int, frame_ms: int = 20) -> Iterable[bytes]:
    """Yield 20ms frames as bytes for WebRTC VAD."""
    audio = _to_int16_pcm(audio)
    frame_len = int(sample_rate * frame_ms / 1000)
    total_frames = len(audio) // frame_len
    for i in range(total_frames):
        start = i * frame_len
        end = start + frame_len
        frame = audio[start:end]
        yield frame.tobytes()


def has_speech(
    audio: np.ndarray, sample_rate: int, aggressiveness: int = 2, min_voiced_ratio: float = 0.1
) -> tuple[bool, float]:
    """Return (has_speech, voiced_ratio). Raises if VAD is unavailable."""
    if webrtcvad is None:
        raise VADUnavailable("webrtcvad is not installed")
    vad = webrtcvad.Vad(aggressiveness)
    frames = list(_frame_generator(audio, sample_rate))
    if not frames:
        return False, 0.0
    voiced = sum(1 for f in frames if vad.is_speech(f, sample_rate))
    ratio = voiced / len(frames)
    logger.debug("VAD voiced ratio=%.3f (voiced=%d total=%d)", ratio, voiced, len(frames))
    return ratio >= min_voiced_ratio, ratio
