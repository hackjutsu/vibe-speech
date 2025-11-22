from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .automation import OutputAutomator
from .config import AppConfig
from .processor import process_text
from .whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)


@dataclass
class RuntimeStatus:
    listening: bool
    last_text: Optional[str]


class SpeechRuntime:
    """Coordinates audio capture, Whisper, processing, and output."""

    def __init__(self, config: AppConfig, model_dir: Optional[Path] = None) -> None:
        self.config = config
        self.model_dir = model_dir
        self.automator = OutputAutomator(config.output)
        self.whisper = WhisperEngine(config.whisper, model_dir=model_dir)
        self._listening = False
        self._last_text: Optional[str] = None

    def start(self) -> None:
        logger.info("Starting speech runtime (stub).")
        self._listening = True
        # TODO: wire microphone streaming + whisper inference loop.
        logger.warning("Audio capture and transcription not implemented yet.")

    def stop(self) -> None:
        logger.info("Stopping speech runtime.")
        self._listening = False

    def run_once(self, text: str) -> RuntimeStatus:
        """Helper for testing: process and send a provided transcript string."""
        processed = process_text(self.config.processing, text)
        result = self.automator.send_text(processed)
        self._last_text = processed if result.sent or result.dry_run else None
        return RuntimeStatus(listening=self._listening, last_text=self._last_text)

    def block_forever(self) -> None:
        """Temporary placeholder main loop."""
        self.start()
        try:
            while self._listening:
                time.sleep(0.25)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.stop()

