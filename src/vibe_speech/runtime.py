from __future__ import annotations

import itertools
import logging
import sys
import threading
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
from .rewriter import LocalLLMRewriter
from .whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)
_COLOR_GREEN = "\033[92m"
_COLOR_CYAN = "\033[96m"
_COLOR_DIM = "\033[90m"
_COLOR_RESET = "\033[0m"


@dataclass
class RuntimeStatus:
    listening: bool
    last_text: Optional[str]


class SpeechRuntime:
    """Coordinates audio capture, Whisper, processing, and output."""

    def __init__(self, config: AppConfig, model_dir: Optional[Path] = None) -> None:
        self.config = config
        cfg_model_dir = Path(config.whisper.model_dir).expanduser() if config.whisper.model_dir else None
        self.model_dir = model_dir or cfg_model_dir
        self.automator = OutputAutomator(config.output)
        self.whisper = WhisperEngine(config.whisper, model_dir=self.model_dir)
        self.rewriter = LocalLLMRewriter(config.rewriter)
        self._listening = False
        self._last_text: Optional[str] = None
        self._last_sent: str = ""
        self._hotkey_listener: Optional[object] = None
        self._stop = False
        self._capture_failures = 0
        self._session_audio: list[np.ndarray] = []

    def start(self) -> None:
        logger.info("Starting speech runtime.")
        self.whisper.load()
        self._listening = False  # start paused until hotkey toggles on
        if self.config.hotkey.push_to_talk:
            logger.info("Push-to-talk armed. Hold the hotkey: %s", self.config.hotkey.toggle)
        else:
            logger.info("Listening is OFF. Toggle with hotkey: %s", self.config.hotkey.toggle)

    def stop(self) -> None:
        self._flush_session_text()
        logger.info("Stopping speech runtime.")
        self._listening = False
        self._stop = True
        if self._hotkey_listener:
            self._hotkey_listener.stop()

    def run_once(self, text: str) -> RuntimeStatus:
        """Helper for testing: process and send a provided transcript string."""
        try:
            processed, rewrite_time = process_text(self.config.processing, text, self.rewriter)
            result = self._deliver_output(processed)
            return RuntimeStatus(listening=self._listening, last_text=self._last_text)
        except Exception as exc:
            logger.error("Processing failed: %s", exc)
            raise

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
                self._session_audio.append(audio)
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
            self._capture_failures = 0
            audio = np.squeeze(audio)
            # Skip near-silent chunks to avoid Whisper hallucinations.
            rms = float(np.sqrt(np.mean(np.square(audio))))
            if rms < self.config.audio.silence_threshold:
                logger.debug("Skipping silent chunk (rms=%.6f < threshold=%.6f)", rms, self.config.audio.silence_threshold)
                return None
            return audio
        except Exception as exc:  # pragma: no cover - device specific
            logger.error("Audio capture failed: %s", exc)
            self._capture_failures += 1
            if self._capture_failures >= 3:
                logger.error(
                    "Disabling listening after repeated audio failures. "
                    "Set audio.device_name in config to a valid input device and restart."
                )
                self._listening = False
            time.sleep(1.0)
            return None

    def _capture_tail_audio(self) -> None:
        """Capture a short tail after release to avoid clipping final words."""
        tail = self.config.audio.tail_padding_seconds
        if tail <= 0:
            return
        samplerate = self.config.audio.sample_rate
        try:
            logger.debug("Recording tail audio: %.2fs @ %d Hz", tail, samplerate)
            audio = sd.rec(
                int(tail * samplerate),
                samplerate=samplerate,
                channels=1,
                dtype="float32",
                device=self.config.audio.device_name,
            )
            sd.wait()
            audio = np.squeeze(audio)
            self._session_audio.append(audio)
        except Exception as exc:  # pragma: no cover - device specific
            logger.error("Tail audio capture failed: %s", exc)

    def _start_hotkey(self) -> None:
        hotkey = self.config.hotkey.toggle
        if not hotkey:
            logger.warning("Hotkey toggle not configured; always listening.")
            return
        if self.config.hotkey.push_to_talk:
            combo_keys = self._combo_keys(hotkey)
            if not combo_keys:
                logger.warning("Hotkey not understood; push-to-talk disabled.")
                return
            pressed: set[object] = set()

            def on_press(key: object) -> None:
                pressed.add(key)
                if combo_keys.issubset(pressed):
                    self._set_listening(True, reason="push-to-talk")

            def on_release(key: object) -> None:
                if key in pressed:
                    pressed.remove(key)
                if self._listening and not combo_keys.issubset(pressed):
                    self._set_listening(False, reason="push-to-talk")

            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
            self._hotkey_listener = listener
            logger.info("Push-to-talk enabled; hold '%s' to listen.", hotkey)
        else:
            combo = self._to_pynput_combo(hotkey)
            self._hotkey_listener = keyboard.GlobalHotKeys({combo: self._toggle_listening})
            self._hotkey_listener.start()
            logger.info("Hotkey '%s' registered for start/stop listening.", hotkey)

    def _set_listening(self, value: bool, reason: str = "hotkey") -> None:
        if self._listening == value:
            return
        self._listening = value
        if self._listening:
            self._capture_failures = 0
            self._session_audio = []
            self._last_sent = ""
        state = "ON" if self._listening else "OFF"
        logger.info("Listening %s (%s)", state, reason)
        if not self._listening:
            self._capture_tail_audio()
            self._flush_session_text()

    def _toggle_listening(self) -> None:
        self._set_listening(not self._listening, reason="toggle")

    def _combo_keys(self, combo: str) -> set[object]:
        mapping = {
            "ctrl": keyboard.Key.ctrl,
            "control": keyboard.Key.ctrl,
            "shift": keyboard.Key.shift,
            "alt": keyboard.Key.alt,
            "option": keyboard.Key.alt,
            "cmd": keyboard.Key.cmd,
            "command": keyboard.Key.cmd,
            "meta": keyboard.Key.cmd,
            "super": keyboard.Key.cmd,
            "space": keyboard.Key.space,
            "enter": keyboard.Key.enter,
            "return": keyboard.Key.enter,
            "tab": keyboard.Key.tab,
            "esc": keyboard.Key.esc,
            "escape": keyboard.Key.esc,
        }
        keys: set[object] = set()
        for part in (p.strip().lower() for p in combo.split("+")):
            key_obj = mapping.get(part)
            if key_obj:
                keys.add(key_obj)
            elif len(part) == 1:
                keys.add(keyboard.KeyCode.from_char(part))
        return keys

    def _to_pynput_combo(self, combo: str) -> str:
        mapping = {
            "ctrl": "<ctrl>",
            "control": "<ctrl>",
            "shift": "<shift>",
            "alt": "<alt>",
            "option": "<alt>",
            "cmd": "<cmd>",
            "command": "<cmd>",
            "meta": "<cmd>",
            "super": "<cmd>",
            "space": "<space>",
            "enter": "<enter>",
            "return": "<enter>",
            "tab": "<tab>",
            "esc": "<esc>",
            "escape": "<esc>",
        }
        parts = [part.strip().lower() for part in combo.split("+")]
        mapped = [mapping.get(part, part) for part in parts]
        return "+".join(mapped)

    def _start_spinner(self, message: str):
        """Start a simple spinner in the terminal; returns a stopper callable."""
        stop_event = threading.Event()
        spinner = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

        def run() -> None:
            while not stop_event.is_set():
                sym = next(spinner)
                sys.stdout.write(f"\r{_COLOR_DIM}{sym} {message}...{_COLOR_RESET}")
                sys.stdout.flush()
                time.sleep(0.1)
            sys.stdout.write("\r")
            sys.stdout.flush()

        thread = threading.Thread(target=run, daemon=True)
        thread.start()

        def stop() -> None:
            stop_event.set()
            thread.join(timeout=1.0)
            sys.stdout.write("\r")
            sys.stdout.flush()

        return stop

    def _deliver_output(self, processed: str):
        text = processed.strip()
        if not text:
            return None
        if self.config.processing.incremental and self._last_sent and text.startswith(self._last_sent):
            to_send = text[len(self._last_sent) :].lstrip()
            if not to_send:
                return None
        else:
            to_send = text
        result = self.automator.send_text(to_send)
        if result.sent or result.dry_run:
            self._last_text = text
            self._last_sent = text
        return result

    def _flush_session_text(self) -> None:
        if not self._session_audio:
            return
        stop_spinner = self._start_spinner("Processing speech")
        audio = np.concatenate(self._session_audio)
        self._session_audio = []
        try:
            t0 = time.perf_counter()
            transcript = self.whisper.transcribe(audio)
            t1 = time.perf_counter()
            full_text, rewrite_time = process_text(self.config.processing, transcript, self.rewriter)
        except Exception as exc:
            stop_spinner()
            logger.error("Transcription/processing failed: %s", exc)
            return
        if not full_text:
            stop_spinner()
            return
        total_transcribe = t1 - t0
        result = self._deliver_output(full_text)
        if result:
            rewrite_str = (
                f"{_COLOR_CYAN}{rewrite_time:.2f}s{_COLOR_RESET}"
                if rewrite_time is not None
                else f"{_COLOR_DIM}n/a{_COLOR_RESET}"
            )
            logger.info(
                "Session transcript: %s | raw: %s | transcribe=%s%.2fs%s rewrite=%s",
                full_text,
                transcript.strip(),
                _COLOR_GREEN,
                total_transcribe,
                _COLOR_RESET,
                rewrite_str,
            )
        stop_spinner()
