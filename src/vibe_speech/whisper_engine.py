from __future__ import annotations

from pathlib import Path
from typing import Optional

# TODO: import faster_whisper when wiring transcription.
# from faster_whisper import WhisperModel

from .config import WhisperConfig


class WhisperEngine:
    """Placeholder for the local Whisper wrapper."""

    def __init__(self, config: WhisperConfig, model_dir: Optional[Path] = None) -> None:
        self.config = config
        self.model_dir = model_dir
        self.model = None

    def load(self) -> None:
        """Load the Whisper model (not implemented)."""
        # Example sketch (disabled):
        # self.model = WhisperModel(
        #     self.config.model_size,
        #     device="auto",
        #     compute_type=self.config.compute_type,
        #     download_root=str(self.model_dir) if self.model_dir else None,
        # )
        raise NotImplementedError("Whisper model loading is not wired yet.")

    def transcribe(self, audio_chunk: bytes) -> str:
        """Run inference on an audio chunk."""
        raise NotImplementedError("Whisper transcription is not wired yet.")

