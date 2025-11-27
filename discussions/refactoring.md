# Refactoring Opportunities (Themes & Priority)

## Architecture / Separation (High)
- `runtime.py` mixes hotkey control, audio buffering, Whisper, assistant, and speech. Extract helpers (e.g., `HotkeyManager`, `SessionRecorder`, `Transcriber`, `Responder/Speaker`) and let `FlowCoordinator` orchestrate.
- Encapsulate hotkey handling (push-to-talk/toggle) into a small class with explicit state reset and debug logging to avoid sticky states and misfires.
- Wrap audio capture into an `AudioSession` that owns `_session_audio`, tail capture, and silence handling so `_flush_session_text` doesn’t juggle assembly and transcription logic.

## Logging / Observability (High)
- Centralize logging setup: configure module levels (e.g., `faster_whisper` at DEBUG, runtime at INFO) via `logging.py` and make it config-driven.
- Use a spinner helper/context that stops before emitting logs to avoid interleaving.
- Keep stage timings at INFO; move transcript/raw and prompts to DEBUG; add concise per-turn summaries at INFO.

## Prompting / Assistant Behavior (High)
- Factor prompt construction and stop tokens into one helper (local/Ollama) with clear constraints: no history echo, persona guidance.
- Add a defensive trim before TTS to strip any `User:`/`Assistant:` labels or quoted history if the model misbehaves.

## Config / CLI (Medium)
- Add docstrings/comments for key config fields (hotkey, audio device, silence thresholds, tail padding) and validate ranges (e.g., `chunk_seconds > 0`).
- Extend CLI: `--debug`, `--no-speech/--no-type`; show hotkey mode and selected audio device at startup.

## Audio Capture / Error Handling (Medium)
- Add retries/backoff and clearer messaging for device errors (e.g., PaMacCore -50).
- Optionally add a `doctor`-style RMS check to guide silence-threshold tuning.

## Testing / Safety Nets (Medium)
- Unit tests for hotkey state machine (press/release combos), prompt builder (no history echo, stop tokens), and processor cleanup/correct helpers.
- Dry-run pipeline test stub for `_flush_session_text` to validate logging and state transitions.

## Code Quality Nits (Low)
- Remove duplicated imports in `processor.py`; add small docstrings/type hints for internal helpers.
- Consider a shared HTTP client wrapper for assistant/rewriter to unify timeouts/retries/error handling.
