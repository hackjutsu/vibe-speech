from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import threading
import time
from typing import Callable, Optional, TypeVar

import numpy as np

from .flow.audio_session import AudioSession
from .flow.hotkey import HotkeyManager
from .flow.indicator import show_listening_banner
from .flow.spinner import Spinner
from .flow.vad import VADUnavailable, has_speech
from .audio_capture import AudioCapture, AudioCaptureError
from .assistant import LLMAssistant
from .config import AppConfig
from .processor import process_text
from .remote_whisper import RemoteWhisperClient, RemoteWhisperError
from .rewriter import LocalLLMRewriter
from .speech_output import SpeechSynthesizer
from .whisper_engine import WhisperEngine

logger = logging.getLogger(__name__)
_DEFAULT_STAGE_TIMEOUT = 90.0
_COLOR_GREEN = "\033[92m"
_COLOR_CYAN = "\033[96m"
_COLOR_BLUE = "\033[94m"
_COLOR_ORANGE = "\033[38;5;208m"
_COLOR_DIM = "\033[90m"
_COLOR_RESET = "\033[0m"
T = TypeVar("T")


@dataclass
class RuntimeStatus:
    listening: bool
    last_text: Optional[str]


class StageAborted(RuntimeError):
    """Raised when a stage needs to abort the current turn."""


class FlowCoordinator:
    """Coordinates audio capture, Whisper, processing, and output."""

    def __init__(self, config: AppConfig, model_dir: Optional[Path] = None) -> None:
        self.config = config
        cfg_model_dir = Path(config.whisper.model_dir).expanduser() if config.whisper.model_dir else None
        self.model_dir = model_dir or cfg_model_dir
        self.audio_capture = AudioCapture(config.audio)
        self._using_remote_whisper = bool(config.whisper.remote_url)
        self.transcriber = (
            RemoteWhisperClient(config.whisper, config.audio.sample_rate)
            if self._using_remote_whisper
            else WhisperEngine(config.whisper, model_dir=self.model_dir)
        )
        self.rewriter = LocalLLMRewriter(config.rewriter)
        self.assistant = LLMAssistant(config.assistant)
        self.speaker = SpeechSynthesizer(config.speech)
        self.hotkeys = HotkeyManager(config.hotkey.toggle, config.hotkey.push_to_talk)
        self.session = AudioSession()
        self._listening = False
        self._last_text: Optional[str] = None
        self._stop = False
        self._stage_timeout = _DEFAULT_STAGE_TIMEOUT
        self._capture_failures = 0

    def start(self) -> None:
        logger.debug("Starting speech flow.")
        self._log_pipeline_overview()
        self.transcriber.load()
        if self._using_remote_whisper:
            logger.info("Using remote Whisper at %s", self.config.whisper.remote_url)
        self._listening = False  # start paused until hotkey toggles on
        if self.config.hotkey.push_to_talk:
            logger.debug("Push-to-talk armed. Hold the hotkey: %s", self.config.hotkey.toggle)
        else:
            logger.debug("Listening is OFF. Toggle with hotkey: %s", self.config.hotkey.toggle)

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
        logger.debug("Listening %s (%s)", state, reason)
        show_listening_banner(self._listening)
        if not self._listening:
            self._capture_tail_audio()
            self._flush_session_text()

    def _log_pipeline_overview(self) -> None:
        """Emit a concise map of the pipeline stages for debugging."""
        outline = [
            "Audio capture: AudioCapture + AudioSession buffering in flow_coordinator.py",
            "Transcription: WhisperEngine.transcribe in whisper_engine.py (loaded at start)"
            if not self._using_remote_whisper
            else "Transcription: RemoteWhisperClient POST /transcribe",
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
                if self.config.audio.vad_enabled:
                    try:
                        voiced, ratio = has_speech(
                            audio,
                            self.config.audio.sample_rate,
                            self.config.audio.vad_aggressiveness,
                            self.config.audio.vad_min_voiced_ratio,
                        )
                        if not voiced:
                            logger.info(
                                "Skipping transcription: no speech detected (voiced ratio=%.3f < %.3f)",
                                ratio,
                                self.config.audio.vad_min_voiced_ratio,
                            )
                            return
                        else:
                            logger.info(
                                "VAD passed: voiced ratio=%.3f >= %.3f (aggr=%d)",
                                ratio,
                                self.config.audio.vad_min_voiced_ratio,
                                self.config.audio.vad_aggressiveness,
                            )
                    except VADUnavailable:
                        logger.info("VAD unavailable; continuing without voice gate.")
                transcript = self._run_stage_with_timeout(
                    "transcription", lambda: self.transcriber.transcribe(audio)
                )
                t1 = time.perf_counter()
                processed_text, rewrite_time = self._run_stage_with_timeout(
                    "processing", lambda: process_text(self.config.processing, transcript, self.rewriter)
                )
                if not processed_text:
                    return
                logger.debug("Transcript (raw): %s", transcript.strip())
                logger.debug("Transcript (processed): %s", processed_text)
                reply_start = time.perf_counter()
                assistant_reply = self._run_stage_with_timeout(
                    "assistant", lambda: self.assistant.respond(processed_text)
                )
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
                speech_result = self._run_stage_with_timeout(
                    "speech", lambda: self.speaker.speak(assistant_reply.text)
                )
                speak_end = time.perf_counter()
            except RemoteWhisperError as exc:
                logger.error("Remote transcription failed: %s", exc)
                logger.info("Waiting for the next input before resuming.")
                return
            except StageAborted as exc:
                logger.error("%s; waiting for the next input before resuming.", exc)
                return
            except Exception as exc:
                logger.error("Transcription/assistant processing failed: %s", exc)
                logger.info("Waiting for the next input before resuming.")
                return
        total_transcribe = t1 - t0
        reply_time = assistant_reply.duration_s if assistant_reply else (reply_end - reply_start)
        speak_time = speak_end - speak_start
        rewrite_str = (
            f"{_COLOR_CYAN}{rewrite_time:.2f}s{_COLOR_RESET}"
            if rewrite_time is not None
            else f"{_COLOR_DIM}n/a{_COLOR_RESET}"
        )
        rewrite_str_plain = f"{rewrite_time:.2f}s" if rewrite_time is not None else "n/a"
        summary_text = processed_text.replace("\n", " ").strip()
        if len(summary_text) > 120:
            summary_text = summary_text[:117] + "..."
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
        logger.debug(
            "Turn summary | text='%s' | transcribe=%.2fs rewrite=%s llm=%.2fs speech=%.2fs spoken=%s",
            summary_text,
            total_transcribe,
            rewrite_str_plain,
            reply_time,
            speak_time,
            speech_result.spoken or speech_result.dry_run,
        )
        self._last_text = processed_text

    def _run_stage_with_timeout(self, name: str, func: Callable[[], T], timeout_s: Optional[float] = None) -> T:
        """Run a stage with a timeout so the loop can't hang indefinitely."""

        timeout = self._stage_timeout if timeout_s is None else timeout_s

        if timeout <= 0:
            return func()

        result: list[T] = []
        error: list[BaseException] = []

        def target() -> None:
            try:
                result.append(func())
            except BaseException as exc:  # pragma: no cover - passthrough
                error.append(exc)

        thread = threading.Thread(target=target, name=f"{name}-stage", daemon=True)
        thread.start()
        thread.join(timeout)
        if thread.is_alive():
            raise StageAborted(f"{name} stage timed out after {timeout:.1f}s")
        if error:
            raise error[0]
        if not result:
            raise StageAborted(f"{name} stage returned no result.")
        return result[0]
