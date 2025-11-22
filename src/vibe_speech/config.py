from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError
from platformdirs import user_config_dir


class AudioConfig(BaseModel):
    sample_rate: int = Field(16000, description="Target audio sample rate for Whisper.")
    chunk_seconds: float = Field(5.0, description="Length of each audio chunk to transcribe.")
    device_name: Optional[str] = Field(
        default=None, description="Preferred input device; None uses system default."
    )
    silence_threshold: float = Field(
        1e-3, description="RMS threshold; chunks with lower energy are treated as silence and skipped."
    )


class WhisperConfig(BaseModel):
    model_size: str = Field("medium", description="Whisper model size (tiny, base, small, medium, large).")
    compute_type: str = Field("int8", description="ct2 compute type, e.g., int8, int8_float16, float16.")
    beam_size: int = Field(5, description="Beam size for decoding.")
    language: Optional[str] = Field(default=None, description="Optional language hint, e.g., 'en'.")
    model_dir: Optional[str] = Field(
        ".cache/huggingface", description="Local cache directory for model downloads."
    )
    offline: bool = Field(True, description="If true, do not attempt network access for models.")


class ProcessingConfig(BaseModel):
    mode: Literal["raw", "cleanup", "summary", "correct"] = "raw"
    incremental: bool = Field(True, description="If true, only type the new delta when text grows.")
    max_chars: int = Field(0, description="Optional truncate to N chars (0 disables).")


class OutputConfig(BaseModel):
    dry_run: bool = Field(True, description="If true, do not type; log instead.")
    typing_delay: float = Field(0.02, description="Delay between keystrokes for pyautogui.")
    focus_target: Optional[str] = Field(default=None, description="Optional app/window hint for focusing.")


class HotkeyConfig(BaseModel):
    toggle: str = Field("ctrl+shift+space", description="Hotkey to start/stop listening.")
    push_to_talk: bool = Field(True, description="If true, listen only while the hotkey is held.")


class AppConfig(BaseModel):
    audio: AudioConfig = AudioConfig()
    whisper: WhisperConfig = WhisperConfig()
    processing: ProcessingConfig = ProcessingConfig()
    output: OutputConfig = OutputConfig()
    hotkey: HotkeyConfig = HotkeyConfig()
    log_level: str = Field("INFO", description="Logging level, e.g., INFO, DEBUG.")

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> "AppConfig":
        path = path or default_config_path()
        if not path.exists():
            return cls()
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        try:
            return cls.model_validate(data)
        except ValidationError as exc:
            raise SystemExit(f"Invalid config at {path}: {exc}") from exc


def default_config_path() -> Path:
    cfg_dir = Path(user_config_dir("vibe-speech"))
    return cfg_dir / "config.yaml"
