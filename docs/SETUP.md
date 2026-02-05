# Setup Guide

This guide covers the three supported modes:

- one-box (record + transcribe + web on one host)
- split machines (capture on one host, transcription/web on another)
- browser capture from phone/tablet/any browser device

## 1) Base Install (all machines)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
transcribe-local setup
transcribe-local doctor
```

Runtime path defaults to `~/.transcribe_local`. Override with:

- `TRANSCRIBE_LOCAL_HOME=/path/to/data-root`
- `TRANSCRIBE_LOCAL_ENV_FILE=/path/to/.env`

## 2) One-Box Mode

```bash
transcribe-local jobs start
transcribe-local serve --host 0.0.0.0 --port 8000
```

Use `http://127.0.0.1:8000`:

- `/` for file upload/transcript management
- `/capture` for browser recording

## 3) Split-Machine Mode

### Server host (Linux/GPU recommended)

```bash
transcribe-local setup --server http://<server-ip>:8000
transcribe-local jobs start
transcribe-local serve --host 0.0.0.0 --port 8000
```

Ensure firewall/security groups allow TCP `8000` from capture clients.

### Capture host (Mac/Windows/Linux)

```bash
transcribe-local setup --server http://<server-ip>:8000
transcribe-local record start
transcribe-local vad start
transcribe-local upload start
```

Pipeline:

1. Recorder writes chunks to `recordings/`
2. VAD moves speech chunks to `ready/`
3. Uploader sends files to server `/api/upload`
4. Server job runner processes from `uploads/`

## 4) Browser Recorder Mode (Phone/Any Device)

1. Run server with `jobs start` and `serve`.
2. Open `http://<server-ip>:8000/capture` on your device browser.
3. Grant microphone permission.
4. Start recording.

Behavior:

- Browser sends chunks continuously.
- Server appends chunks into a temporary capture file.
- On stop, server finalizes the audio file into `uploads/`.
- Job runner processes it asynchronously.

## 5) Configuration Quick Reference

- `~/.transcribe_local/config.json`
  - `transcription_mode`
  - `openai_model`
  - `local_model`
  - `default_language`
  - `server_url`
- `~/.transcribe_local/.env`
  - `OPENAI_API_KEY`
  - `HF_TOKEN`

## 6) Troubleshooting

- `doctor` fails on ffmpeg/ffprobe:
  - install ffmpeg and retry `transcribe-local doctor`.
- uploads are accepted but nothing is transcribed:
  - confirm `transcribe-local jobs status` on the server host.
- capture device cannot connect:
  - verify URL/IP/port and network/firewall reachability.
- browser capture unavailable:
  - use a browser with `MediaRecorder` support or use CLI recorder+uploader.
