# TODO and Progress

## Done
- Whisper wiring via faster-whisper with configurable model/beam/compute type.
- Hotkey toggle (push-to-talk), audio capture with tail padding, buffering, and single-shot transcription on release.
- Optional grammar rewriter (Ollama/local llama.cpp); processing modes (`raw`, cleanup, correct).
- Spinner and colored timing logs for transcribe/rewrite; tail capture to reduce clipping.
- Config expanded: silence gate, tail padding, initial prompt for Whisper, rewriter backend/options.
- Docs updated (README); sample config expanded.

## Next
- Add streaming/partial output and/or VAD-based end-of-speech for lower latency.
- Per-OS focus helpers (Hammerspoon/AppleScript, AutoHotkey) beyond pyautogui.
- Tests for audio/processing/rewriter paths.
- UX polish: better in-app indicators, config validation, and device selection helper.
