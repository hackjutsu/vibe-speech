# TODO and Progress

## Done
- Created initial scaffold: `pyproject.toml`, `README.md`, `config.sample.yaml`, and package skeleton under `src/vibe_speech/` (config, CLI, runtime stub, automation stub, processor, whisper placeholder).
- CLI commands added: `serve` (stub loop, dry-run by default) and `doctor` (inspect config).
- Basic config handling (YAML + defaults) and dry-run typing via `pyautogui`.

## Next
- Wire microphone capture (`sounddevice`) with chunked streaming and pass into Whisper.
- Implement WhisperEngine using `faster-whisper` (medium model, float16 on Apple Silicon) with local model cache.
- Add hotkey toggle and simple state machine (likely `pynput`) to start/stop listening.
- Implement per-OS focus/typing helpers (Hammerspoon/AppleScript on macOS, AutoHotkey on Windows) with pyautogui fallback; keep dry-run safety switch.
- Enhance text processing: configurable cleanup/summary step; add truncation safeguards.
- Add tests for config and text processing; expand coverage as audio/automation land.
- Improve docs: platform-specific setup, troubleshooting for audio devices and model downloads.
