# Remote Whisper Service (Compatible with Current Client)

Goal: run Whisper on a more powerful remote machine while keeping the current client flow. The client already buffers audio during a listening window and transcribes once on hotkey release. We send the concatenated session audio in a single request to the remote service.

## Service Shape (Minimal)
- HTTP endpoint: `POST /transcribe`
- Request (JSON):
  - `audio` (base64 of mono 16 kHz int16 PCM)
  - `sample_rate` (e.g., 16000)
  - Optional: `language`, `initial_prompt`, `beam_size`, `compute_type`
- Response (JSON):
  - `text` (transcript)
  - Optional: `info` (timings, language detected)
- Transport: HTTP/JSON with gzip; base64 audio payload kept small by using int16 PCM.

## Why One-Shot (for Compatibility)
- Current client only transcribes at stop: chunks + tail → concatenate → single call.
- No mid-stream or partial transcripts in the client today; one-shot keeps client changes minimal.
- If streaming is desired later, add a streaming API and do a final full-context decode at segment end to fix partials.

## Server Implementation Sketch
- Stack: Python + FastAPI (or Flask) + `faster-whisper`.
- Load model once at startup (`whisper-large-v3-turbo` or similar); keep in memory.
- Endpoint steps:
  1) Decode base64 to int16 PCM; verify sample rate.
  2) Run `transcribe(audio, beam_size=?, language=?, initial_prompt=?, compute_type=?)`.
  3) Return `{ "text": transcript, "info": { timings, language } }`.
- Optional: accept `compute_type` (e.g., float16/int8) to match GPU/CPU capabilities.

## Client Integration (Changes Needed)
- Add config: `whisper.remote_url` (when set, use remote service).
- Add `RemoteWhisperClient` to POST `{audio, sample_rate, ...}` and parse JSON.
- In `_flush_session_text`, if `remote_url` is set, call remote; else use local `WhisperEngine`.
- Keep local VAD/silence gating to avoid sending empty/noise audio.
- Handle errors with a clear log and optional fallback to local (configurable).

## Payload Details
- Audio format: mono, 16 kHz, int16 PCM (little-endian). Matches current capture config.
- Size: small per session (chunk_seconds ~2s buffered over a hotkey window + tail).
- Compression: enable gzip on request to reduce payload.

## Security & Deployment
- Run on LAN (e.g., `http://<local ip>:<port>/transcribe`).
- Add a simple auth header/token if exposure is a concern; restrict to LAN IPs.
- Deployment: systemd service or Docker container on the GPU box; warm model at startup.

## Future Streaming Option
- If lower latency is needed: add a streaming endpoint that accepts chunked audio with overlap/context and returns partials. Still run a final full-context decode when the segment ends to correct errors.
