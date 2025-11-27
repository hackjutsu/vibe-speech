from __future__ import annotations

import logging
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .flow.audio_session import AudioSession
from .flow.hotkey import HotkeyManager
from .flow.spinner import Spinner
from .audio_capture import AudioCapture, AudioCaptureError
from .assistant import LLMAssistant
from .config import AppConfig
from .processor import process_text
from .rewriter import LocalLLMRewriter
from .speech_output import SpeechSynthesizer
from .whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)
_COLOR_GREEN = "\033[92m"
_COLOR_CYAN = "\033[96m"
_COLOR_BLUE = "\033[94m"
_COLOR_ORANGE = "\033[38;5;208m"
_COLOR_DIM = "\033[90m"
_COLOR_RESET = "\033[0m"


@dataclass
class RuntimeStatus:
    listening: bool
    last_text: Optional[str]


class FlowCoordinator:
    """Coordinates audio capture, Whisper, processing, and output."""

    def __init__(self, config: AppConfig, model_dir: Optional[Path] = None) -> None:
        self.config = config
        cfg_model_dir = Path(config.whisper.model_dir).expanduser() if config.whisper.model_dir else None
        self.model_dir = model_dir or cfg_model_dir
        self.audio_capture = AudioCapture(config.audio)
        self.whisper = WhisperEngine(config.whisper, model_dir=self.model_dir)
        self.rewriter = LocalLLMRewriter(config.rewriter)
        self.assistant = LLMAssistant(config.assistant)
        self.speaker = SpeechSynthesizer(config.speech)
        self.hotkeys = HotkeyManager(config.hotkey.toggle, config.hotkey.push_to_talk)
        self.session = AudioSession()
        self._listening = False
        self._last_text: Optional[str] = None
        self._stop = False
        self._capture_failures = 0

    def start(self) -> None:
        logger.info("Starting speech flow.")
        self._log_pipeline_overview()
        self.whisper.load()
        self._listening = False  # start paused until hotkey toggles on
        if self.config.hotkey.push_to_talk:
            logger.info("Push-to-talk armed. Hold the hotkey: %s", self.config.hotkey.toggle)
        else:
            logger.info("Listening is OFF. Toggle with hotkey: %s", self.config.hotkey.toggle)

    def stop(self) -> None:
        self._flush_session_text()
        logger.info("Stopping speech flow.")
        self._listening = False
        self._stop = True
        self.hotkeys.stop()

    def run_once(self, text: str) -> RuntimeStatus:
        """Helper for testing: process and send a provided transcript string."""
        try:
            processed, _ = process_text(self.config.processing, text, self.rewriter)
            reply = self.assistant.respond(processed)
            self.speaker.speak(reply.text)
            self._last_text = processed
            return RuntimeStatus(listening=self._listening, last_text=self._last_text)
        except Exception as exc:
            logger.error("Processing failed: %s", exc)
            raise

    def block_forever(self) -> None:
        """Main loop: capture audio chunks, transcribe, and send output."""
        self.start()
        self.hotkeys.start(self._listen_on, self._listen_off)
        try:
            while not self._stop:
                if not self._listening:
                    time.sleep(0.2)
                    continue
                audio = self._capture_chunk()
                if audio is None:
                    continue
                self.session.append(audio)
        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self.stop()

    def _listen_on(self) -> None:
        self._set_listening(True, reason="hotkey")

    def _listen_off(self) -> None:
        self._set_listening(False, reason="hotkey")

    def _capture_chunk(self) -> Optional[np.ndarray]:
        """Record a chunk of audio and return a numpy array."""
        try:
            audio = self.audio_capture.capture_chunk()
            self._capture_failures = 0
            return audio
        except AudioCaptureError as exc:  # pragma: no cover - device specific
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
        try:
            tail = self.audio_capture.capture_tail()
        except AudioCaptureError as exc:  # pragma: no cover - device specific
            logger.error("Tail audio capture failed: %s", exc)
            return
        if tail is not None:
            self.session.append(tail)

    def _set_listening(self, value: bool, reason: str = "hotkey") -> None:
        if self._listening == value:
            return
        self._listening = value
        if self._listening:
            self._capture_failures = 0
            self.session.reset()
        state = "ON" if self._listening else "OFF"
        logger.info("Listening %s (%s)", state, reason)
        if not self._listening:
            self._capture_tail_audio()
            self._flush_session_text()

    def _log_pipeline_overview(self) -> None:
        """Emit a concise map of the pipeline stages for debugging."""
        outline = [
            "Audio capture: AudioCapture + AudioSession buffering in flow_coordinator.py",
            "Transcription: WhisperEngine.transcribe in whisper_engine.py (loaded at start)",
            "Text processing: process_text + optional LocalLLMRewriter in processor.py/rewriter.py",
            "Assistant reply: LLMAssistant.respond in assistant.py",
            "Output: SpeechSynthesizer.speak in speech_output.py; typing automation via automation.py",
            "Orchestration: FlowCoordinator in flow_coordinator.py; CLI entry in cli.py",
        ]
        for stage in outline:
            logger.debug("Pipeline stage -> %s", stage)

    def _flush_session_text(self) -> None:
        audio = self.session.consume()
        if audio is None:
            return
        with Spinner("Processing speech"):
            try:
                t0 = time.perf_counter()
                transcript = self.whisper.transcribe(audio)
                t1 = time.perf_counter()
                processed_text, rewrite_time = process_text(self.config.processing, transcript, self.rewriter)
                if not processed_text:
                    return
                logger.debug("Transcript (raw): %s", transcript.strip())
                logger.debug("Transcript (processed): %s", processed_text)
                reply_start = time.perf_counter()
                assistant_reply = self.assistant.respond(processed_text)
                reply_end = time.perf_counter()
                user_block = (
                    f"\n\n{_COLOR_BLUE}user input(you):{_COLOR_RESET}\n"
                    f"{_COLOR_BLUE}{processed_text}{_COLOR_RESET}"
                )
                assistant_block = (
                    f"{_COLOR_ORANGE}assistant response:{_COLOR_RESET}\n"
                    f"{_COLOR_ORANGE}{assistant_reply.text.strip()}{_COLOR_RESET}"
                )
                logger.info("%s\n\n%s\n", user_block, assistant_block)
                speak_start = time.perf_counter()
                speech_result = self.speaker.speak(assistant_reply.text)
                speak_end = time.perf_counter()
            except Exception as exc:
                logger.error("Transcription/assistant processing failed: %s", exc)
                return
        total_transcribe = t1 - t0
        reply_time = assistant_reply.duration_s if assistant_reply else (reply_end - reply_start)
        speak_time = speak_end - speak_start
        rewrite_str = (
            f"{_COLOR_CYAN}{rewrite_time:.2f}s{_COLOR_RESET}"
            if rewrite_time is not None
            else f"{_COLOR_DIM}n/a{_COLOR_RESET}"
        )
        logger.info(
            "Stage timings:\n"
            "  Transcribe: %s%.2fs%s\n"
            "  Rewrite:    %s\n"
            "  Assistant:  %s%.2fs%s\n"
            "  Speech:     %s%.2fs%s",
            _COLOR_GREEN,
            total_transcribe,
            _COLOR_RESET,
            rewrite_str,
            _COLOR_CYAN,
            reply_time,
            _COLOR_RESET,
            _COLOR_CYAN,
            speak_time,
            _COLOR_RESET,
        )
        self._last_text = processed_text
