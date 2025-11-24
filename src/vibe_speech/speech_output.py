from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass

from .config import SpeechOutputConfig

logger = logging.getLogger(__name__)


@dataclass
class SpeechResult:
    spoken: bool
    dry_run: bool


class SpeechSynthesizer:
    """Send text to an external TTS command (default: xsst2)."""

    def __init__(self, config: SpeechOutputConfig) -> None:
        self.config = config

    def speak(self, text: str) -> SpeechResult:
        cleaned = text.strip()
        if not cleaned:
            logger.debug("Skipping empty assistant reply.")
            return SpeechResult(spoken=False, dry_run=self.config.dry_run)

        if not self.config.enabled:
            logger.info("Speech output disabled; reply: %s", cleaned)
            return SpeechResult(spoken=False, dry_run=self.config.dry_run)

        if self.config.dry_run:
            logger.info("[dry-run] Assistant reply: %s", cleaned)
            return SpeechResult(spoken=False, dry_run=True)

        cmd = [self.config.command, *self.config.extra_args]
        try:
            subprocess.run(cmd, input=cleaned.encode("utf-8"), check=True)
            return SpeechResult(spoken=True, dry_run=False)
        except Exception as exc:  # pragma: no cover - platform-specific
            logger.error("Failed to invoke speech output (%s): %s", self.config.command, exc)
            return SpeechResult(spoken=False, dry_run=self.config.dry_run)
