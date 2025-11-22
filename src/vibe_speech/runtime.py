from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from pynput import keyboard

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
        self._hotkey_listener: Optional[keyboard.GlobalHotKeys] = None
        self._stop = False

    def start(self) -> None:
        logger.info("Starting speech runtime.")
        self.whisper.load()
        self._listening = True

    def stop(self) -> None:
        logger.info("Stopping speech runtime.")
        self._listening = False
        self._stop = True
        if self._hotkey_listener:
            self._hotkey_listener.stop()

    def run_once(self, text: str) -> RuntimeStatus:
        """Helper for testing: process and send a provided transcript string."""
        processed = process_text(self.config.processing, text)
        result = self.automator.send_text(processed)
        self._last_text = processed if result.sent or result.dry_run else None
        return RuntimeStatus(listening=self._listening, last_text=self._last_text)

    def block_forever(self) -> None:
        """Main loop: capture audio chunks, transcribe, and send output."""
        self.start()
        self._start_hotkey()
        try:
            while not self._stop:
                if not self._listening:
                    time.sleep(0.2)
                    continue
                audio = self._capture_chunk()
                if audio is None:
                    continue
                try:
                    transcript = self.whisper.transcribe(audio)
                except Exception as exc:  # pragma: no cover - runtime guard
                    logger.error("Transcription failed: %s", exc)
                    continue
                processed = process_text(self.config.processing, transcript)
                result = self.automator.send_text(processed)
                self._last_text = processed if result.sent or result.dry_run else None
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.stop()

    def _capture_chunk(self) -> Optional[np.ndarray]:
        """Record a chunk of audio and return a numpy array."""
        duration = self.config.audio.chunk_seconds
        samplerate = self.config.audio.sample_rate
        try:
            logger.debug("Recording audio chunk: %.2fs @ %d Hz", duration, samplerate)
            audio = sd.rec(
                int(duration * samplerate),
                samplerate=samplerate,
                channels=1,
                dtype="float32",
                device=self.config.audio.device_name,
            )
            sd.wait()
            return np.squeeze(audio)
        except Exception as exc:  # pragma: no cover - device specific
            logger.error("Audio capture failed: %s", exc)
            time.sleep(1.0)
            return None

    def _start_hotkey(self) -> None:
        hotkey = self.config.hotkey.toggle
        if not hotkey:
            logger.warning("Hotkey toggle not configured; always listening.")
            return

        combo = self._to_pynput_combo(hotkey)
        self._hotkey_listener = keyboard.GlobalHotKeys({combo: self._toggle_listening})
        self._hotkey_listener.start()
        logger.info("Hotkey '%s' registered for start/stop listening.", hotkey)

    def _toggle_listening(self) -> None:
        self._listening = not self._listening
        state = "ON" if self._listening else "OFF"
        logger.info("Listening toggled %s", state)

    def _to_pynput_combo(self, combo: str) -> str:
        parts = [part.strip().lower() for part in combo.split("+")]
        mapped = []
        for part in parts:
            if part in {"ctrl", "control"}:
                mapped.append("<ctrl>")
            elif part in {"shift"}:
                mapped.append("<shift>")
            elif part in {"alt", "option"}:
                mapped.append("<alt>")
            elif part in {"cmd", "command", "meta", "super"}:
                mapped.append("<cmd>")
            else:
                mapped.append(part)
        return "+".join(mapped)
