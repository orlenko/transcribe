# transcribe-local

Speaker-aware transcription toolkit with:

- OpenAI or local transcription modes
- Local diarization and speaker matching
- FastAPI web UI for upload/review/edit/export
- Background job runner for asynchronous processing
- Recorder/VAD/uploader pipeline for distributed setups
- Browser recorder page (`/capture`) for phone/any-device input

## Installation

### Prerequisites

- Python `3.10+`
- `ffmpeg` and `ffprobe` in `PATH`
- Hugging Face token (`HF_TOKEN`) for local diarization (`hybrid`/`local`)
- OpenAI API key (`OPENAI_API_KEY`) for `hybrid`/`openai`

### Install

```bash
git clone https://github.com/orlenko/transcribe.git
cd transcribe
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## First Run

```bash
transcribe-local setup
transcribe-local doctor
```

This writes runtime state under `~/.transcribe_local/`:

- `config.json`
- `.env`
- `uploads/`, `processed/`, `recordings/`, `ready/`, `uploaded/`
- `speakers.db`

## Deployment Topologies

### 1) One machine (all-in-one)

```bash
transcribe-local jobs start
transcribe-local serve --host 0.0.0.0 --port 8000
```

Open `http://<host>:8000` and upload audio files.

### 2) Two machines (recommended for heavy models)

- Machine A (Linux/GPU server): run `jobs` + `serve`.
- Machine B (capture client): run recorder/VAD/uploader pointing to server URL.

Server machine:

```bash
transcribe-local setup --server http://<server-ip>:8000
transcribe-local jobs start
transcribe-local serve --host 0.0.0.0 --port 8000
```

Capture machine:

```bash
transcribe-local setup --server http://<server-ip>:8000
transcribe-local record start
transcribe-local vad start
transcribe-local upload start
```

`upload start` uses configured `server_url` by default.

### 3) Browser/phone recorder to server

Run the server on a reachable host, then open:

`http://<server-ip>:8000/capture`

The page uses `MediaRecorder` and uploads audio chunks continuously to the server.
When you stop recording, the assembled file is queued in `uploads/` for the job runner.

## Common Commands

```bash
transcribe-local setup
transcribe-local doctor
transcribe-local serve --host 0.0.0.0 --port 8000
transcribe-local jobs start
transcribe-local upload start
transcribe-local transcribe ./sample.mp3 --hybrid
```

## Notes

- Processing mode/language for queued files is controlled by server/job-runner configuration.
- `TRANSCRIBE_LOCAL_HOME` changes the runtime root (default `~/.transcribe_local`).
- `TRANSCRIBE_LOCAL_ENV_FILE` points to a custom `.env` file.
- Legacy helpers (`transcribe.py`, `transcribe_batch.sh`) are still included.
