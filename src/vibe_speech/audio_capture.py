from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import sounddevice as sd

from .config import AudioConfig

logger = logging.getLogger(__name__)


class AudioCaptureError(Exception):
    """Raised when audio capture fails."""


class AudioCapture:
    """Records mic input and filters silent chunks."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config

    def capture_chunk(self) -> Optional[np.ndarray]:
        duration = self.config.chunk_seconds
        samplerate = self.config.sample_rate
        try:
            logger.debug("Recording audio chunk: %.2fs @ %d Hz", duration, samplerate)
            audio = sd.rec(
                int(duration * samplerate),
                samplerate=samplerate,
                channels=1,
                dtype="float32",
                device=self.config.device_name,
            )
            sd.wait()
        except Exception as exc:  # pragma: no cover - device specific
            raise AudioCaptureError(exc) from exc

        audio = self._sanitize(audio)
        if audio is None:
            return None

        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < self.config.silence_threshold:
            logger.debug(
                "Skipping silent chunk (rms=%.6f < threshold=%.6f)", rms, self.config.silence_threshold
            )
            return None
        return audio

    def capture_tail(self) -> Optional[np.ndarray]:
        tail = self.config.tail_padding_seconds
        if tail <= 0:
            return None
        samplerate = self.config.sample_rate
        try:
            logger.debug("Recording tail audio: %.2fs @ %d Hz", tail, samplerate)
            audio = sd.rec(
                int(tail * samplerate),
                samplerate=samplerate,
                channels=1,
                dtype="float32",
                device=self.config.device_name,
            )
            sd.wait()
        except Exception as exc:  # pragma: no cover - device specific
            raise AudioCaptureError(exc) from exc

        return self._sanitize(audio)

    def _sanitize(self, audio: np.ndarray) -> Optional[np.ndarray]:
        """Drop obviously bad audio buffers before downstream processing."""
        audio = np.squeeze(audio).astype(np.float32, copy=False)
        if audio.size == 0:
            logger.warning("Dropping empty audio buffer from device=%s", self.config.device_name or "default")
            return None
        if not np.all(np.isfinite(audio)):
            logger.warning("Dropping chunk with non-finite samples from device=%s", self.config.device_name or "default")
            return None
        max_abs = float(np.max(np.abs(audio)))
        if max_abs > 10:  # far beyond normalized PCM range; likely device error
            logger.warning(
                "Dropping chunk with suspicious amplitude (max=%.2f) from device=%s",
                max_abs,
                self.config.device_name or "default",
            )
            return None
        return audio
