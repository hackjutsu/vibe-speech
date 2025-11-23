# Vibe Speech (WIP)

Cross-platform, local-first voice-to-text helper that listens to the mic, runs Whisper locally, and types the result into the focused terminal or editor. Designed for macOS and Windows, with a fallback automation layer that works anywhere `pyautogui` does.

## Current status
- Push-to-talk hotkey with tail padding; buffers while held, then transcribes once and types the full result.
- Faster-Whisper backend (configurable model; defaults to `large-v3-turbo` unless you change it). Beam size, language, and an optional initial prompt are configurable.
- Optional rewriter (Ollama or llama.cpp) for grammar polish; can be disabled for raw transcripts.
- Spinner/colored timing logs so you can see when transcription/rewriting is running.

## Project layout
- `src/vibe_speech/cli.py` – entrypoint (`vibe-speech` script) with `serve` and `doctor`.
- `src/vibe_speech/config.py` – config models and loader.
- `src/vibe_speech/runtime.py` – audio capture, hotkey handling, buffering, transcription, output, and logging.
- `src/vibe_speech/whisper_engine.py` – Whisper wrapper (faster-whisper).
- `src/vibe_speech/automation.py` – text output automation (pyautogui).
- `src/vibe_speech/processor.py` – processing modes (`raw`, cleanup, optional correction/rewriter).
- `src/vibe_speech/rewriter.py` – optional grammar rewriter (Ollama/local llama.cpp).
- `config.sample.yaml` – defaults for local development.

## Quick start
1) Python 3.11+, `ffmpeg` on PATH.
2) Install: `python -m venv .venv && source .venv/bin/activate && pip install -e .`
3) Copy and edit config: `cp config.sample.yaml config.yaml`
   - Set `audio.device_name` to your mic.
   - Choose a Whisper model (`whisper.model_size`), beam size, `initial_prompt` if desired.
   - Enable/disable the rewriter as needed.
4) Run: `vibe-speech --config config.yaml serve` (use `--dry-run` to avoid typing).
5) Hold `ctrl+shift+space` (default) while speaking; release to transcribe and type. Spinner shows work in progress; logs include transcribe/rewrite timing and raw/final text.

## Platform notes
- macOS: grant Accessibility for your terminal/editor so typing works; mic permission for the terminal. Tail padding helps avoid clipping; adjust in config.
- Windows/Linux: relies on `pyautogui` for typing; focus targeting not yet implemented.

## Troubleshooting
- Mic errors (AUHAL -50, etc.): set `audio.device_name` to a valid input from `sounddevice.query_devices()`, and ensure mic permission is granted.
- Model downloads: set `whisper.offline: false` for the first run to cache; then flip to `true` for offline use.
- Accuracy vs speed: smaller models/beam=1–3 for speed; larger/beam=5 for accuracy.

## Notes
- Streaming/partials are not implemented; the app buffers until hotkey release (with optional tail capture).
- The rewriter can change phrasing; set `processing.mode: raw` and `rewriter.enabled: false` for unaltered Whisper output.
