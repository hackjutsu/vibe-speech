from __future__ import annotations

from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

from .config import WhisperConfig


class WhisperEngine:
    """Placeholder for the local Whisper wrapper."""

    def __init__(self, config: WhisperConfig, model_dir: Optional[Path] = None) -> None:
        self.config = config
        self.model_dir = model_dir
        self.model: Optional[WhisperModel] = None

    def load(self) -> None:
        """Load the Whisper model."""
        self.model = WhisperModel(
            self.config.model_size,
            device="auto",
            compute_type=self.config.compute_type,
            download_root=str(self.model_dir) if self.model_dir else None,
        )

    def transcribe(self, audio_chunk) -> str:  # audio_chunk: np.ndarray or path-like
        """Run inference on an audio chunk."""
        if self.model is None:
            raise RuntimeError("Whisper model not loaded. Call load() first.")

        segments, _info = self.model.transcribe(
            audio_chunk,
            beam_size=self.config.beam_size,
            language=self.config.language,
        )
        text_parts = [segment.text.strip() for segment in segments if segment.text]
        return " ".join(text_parts).strip()
