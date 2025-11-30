from __future__ import annotations

import base64
import logging
from typing import Optional

import numpy as np
import requests

from .config import WhisperConfig

logger = logging.getLogger(__name__)


class RemoteWhisperError(RuntimeError):
    """Raised when the remote Whisper call fails."""


class RemoteWhisperClient:
    """Simple client for the remote Whisper server described in discussions/remote-whisper-server.md."""

    def __init__(self, config: WhisperConfig, sample_rate: int) -> None:
        if not config.remote_url:
            raise ValueError("remote_url must be set for RemoteWhisperClient")
        self.config = config
        self.sample_rate = sample_rate
        self.endpoint = self._make_endpoint(config.remote_url)

    def _make_endpoint(self, base_url: str) -> str:
        return base_url.rstrip("/") + "/transcribe"

    def load(self) -> None:  # for parity with the local WhisperEngine
        return None

    def transcribe(self, audio: np.ndarray) -> str:
        if audio is None:
            raise RemoteWhisperError("No audio provided for remote transcription.")

        float_audio = audio.astype(np.float32, copy=False)
        clipped = np.clip(float_audio, -1.0, 1.0)
        pcm16 = (clipped * 32767.0).astype(np.int16)
        payload = self._build_payload(pcm16)

        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.config.remote_timeout,
                headers={"Accept": "application/json"},
            )
            response.raise_for_status()
        except Exception as exc:  # pragma: no cover - network/remote specific
            raise RemoteWhisperError(f"Remote Whisper request failed: {exc}") from exc

        try:
            data = response.json()
        except Exception as exc:  # pragma: no cover - network/remote specific
            raise RemoteWhisperError(f"Failed to parse Whisper response JSON: {exc}") from exc

        text = (data.get("text") or "").strip()
        if not text:
            logger.warning("Remote Whisper returned an empty transcript.")
        return text

    def _build_payload(self, pcm16: np.ndarray) -> dict[str, object]:
        audio_bytes = pcm16.tobytes()
        audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        payload: dict[str, object] = {
            "audio": audio_b64,
            "sample_rate": self.sample_rate,
        }

        if self.config.language:
            payload["language"] = self.config.language
        if self.config.initial_prompt:
            payload["initial_prompt"] = self.config.initial_prompt
        if self.config.beam_size:
            payload["beam_size"] = self.config.beam_size
        if self.config.compute_type:
            payload["compute_type"] = self.config.compute_type

        return payload
