from __future__ import annotations

import logging
from dataclasses import dataclass

import pyautogui

from .config import OutputConfig

logger = logging.getLogger(__name__)


@dataclass
class AutomationResult:
    sent: bool
    dry_run: bool


class OutputAutomator:
    """Types text into the focused window. Falls back to dry-run logging."""

    def __init__(self, config: OutputConfig) -> None:
        self.config = config
        pyautogui.PAUSE = self.config.typing_delay

    def send_text(self, text: str) -> AutomationResult:
        text = text.rstrip()
        if not text:
            logger.debug("Skipping empty transcription.")
            return AutomationResult(sent=False, dry_run=self.config.dry_run)

        if self.config.dry_run:
            logger.info("[dry-run] Would type: %s", text)
            return AutomationResult(sent=False, dry_run=True)

        try:
            pyautogui.typewrite(text)
            return AutomationResult(sent=True, dry_run=False)
        except Exception as exc:  # pragma: no cover - platform-specific
            logger.error("Failed to type text: %s", exc)
            return AutomationResult(sent=False, dry_run=self.config.dry_run)

