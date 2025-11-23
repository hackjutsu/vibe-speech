from __future__ import annotations

import os
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
        """Load the Whisper model with a compatibility fallback."""
        compute_order = [self.config.compute_type]
        if "int8" not in compute_order:
            compute_order.append("int8")
        if "float32" not in compute_order:
            compute_order.append("float32")

        last_error: Optional[Exception] = None
        if self.config.offline:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if self.model_dir:
            os.environ.setdefault("HF_HOME", str(self.model_dir))

        for compute in compute_order:
            try:
                self.model = WhisperModel(
                    self._model_source(),
                    device="auto",
                    compute_type=compute,
                    download_root=str(self.model_dir) if self.model_dir else None,
                )
                self.config.compute_type = compute  # record the working type
                return
            except ValueError as exc:
                last_error = exc
                continue
        if last_error:
            raise last_error
        raise RuntimeError("Failed to load Whisper model for unknown reasons.")

    def _model_source(self) -> str:
        if self.model_dir:
            path = Path(self.model_dir)
            if path.exists():
                return self.config.model_size  # rely on HF cache in model_dir
        return self.config.model_size

    def transcribe(self, audio_chunk) -> str:  # audio_chunk: np.ndarray or path-like
        """Run inference on an audio chunk."""
        if self.model is None:
            raise RuntimeError("Whisper model not loaded. Call load() first.")

        segments, _info = self.model.transcribe(
            audio_chunk,
            beam_size=self.config.beam_size,
            language=self.config.language,
            initial_prompt=self.config.initial_prompt,
        )
        text_parts = [segment.text.strip() for segment in segments if segment.text]
        return " ".join(text_parts).strip()
