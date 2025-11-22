# Vibe Speech (WIP)

Cross-platform, local-first voice-to-text helper that listens to the mic, runs Whisper locally, and types the result into the focused terminal or editor. Designed for macOS and Windows, with a fallback automation layer that works anywhere `pyautogui` does.

## Current status
- Repo is scaffolding only; audio capture, Whisper inference, and automation are stubbed.
- Local Whisper (intended: `faster-whisper` with medium-sized model) to fit comfortably on an M4 Pro 24 GB machine.

## Goals
- Hands-free toggle to start/stop listening.
- Choose processing mode: `raw` transcription, light cleanup, or short summary.
- Safe defaults: dry-run output until the user confirms automation works on their machine.
- Configurable hotkeys, typing speed, and target application focus.

## Project layout
- `src/vibe_speech/cli.py` – entrypoint (`vibe-speech` script) with `serve` and `doctor` commands.
- `src/vibe_speech/config.py` – config models and loader (YAML + env overrides).
- `src/vibe_speech/runtime.py` – orchestrates audio, Whisper, processing, and output (currently stubbed).
- `src/vibe_speech/whisper_engine.py` – placeholder for loading/running local Whisper.
- `src/vibe_speech/automation.py` – text output automation (dry-run by default).
- `src/vibe_speech/processor.py` – text cleanup/summarization placeholder.
- `config.sample.yaml` – defaults for local development.

## Quick start (scaffold)
1) Python 3.11+, `ffmpeg` installed and on PATH (required by Whisper).
2) Create a venv and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
3) Copy config and edit to match your mic and preferences:
   ```bash
   cp config.sample.yaml config.yaml
   ```
4) Run a dry-run service (no typing will occur):
   ```bash
   vibe-speech serve --config config.yaml --dry-run
   ```
   Expect a warning that audio/Whisper loops are not wired yet; this is intentional until implementation lands.

## Platform notes
- macOS: plan to use Hammerspoon or AppleScript for focusing windows; `pyautogui` for typing fallback.
- Windows: plan to use AutoHotkey for focus/typing if available; `pyautogui` fallback.
- Linux: rely on `pyautogui` and window manager shortcuts; may add xdotool integration later.

## Next milestones
- Wire microphone capture (likely `sounddevice`) with chunked streaming.
- Load local Whisper via `faster-whisper`, support medium and small models; compute type settable (`float16` on Apple Silicon).
- Implement hotkey toggle (likely `pynput`) and integrate a small state machine.
- Add focus helpers per-OS (Hammerspoon/AppleScript, AutoHotkey) with a dry-run safety switch.
- Ship tests for text processing and config handling.

## Troubleshooting
- If `pip install -e .` fails on `faster-whisper`, ensure `cmake` and `rust` toolchain are installed; alternatively swap to `openai-whisper` in `pyproject.toml`.
- Audio device selection is config-driven; set `audio.device_name` to match `sounddevice.query_devices()`.

