# transcribe_local Package Guide

This package powers the `transcribe-local` CLI.

For project installation and quick start, see the repository root README.
Detailed deployment playbooks are in `docs/SETUP.md`.

## First-time setup

```bash
transcribe-local setup
transcribe-local doctor
```

The setup command creates and configures:

- `~/.transcribe_local/config.json`
- `~/.transcribe_local/.env`
- Runtime folders (`recordings`, `ready`, `uploaded`, `uploads`, `processed`)

You can override the default data root with:

```bash
export TRANSCRIBE_LOCAL_HOME=/custom/transcribe-data
```

## Main command groups

```bash
transcribe-local transcribe ...
transcribe-local speakers ...
transcribe-local transcripts ...
transcribe-local export ...
transcribe-local serve ...
transcribe-local record ...
transcribe-local vad ...
transcribe-local upload ...
transcribe-local jobs ...
```

## Typical operations

### One-off transcription

```bash
transcribe-local transcribe ./audio.mp3 --hybrid
```

### Start background processing and web UI

```bash
transcribe-local jobs start
transcribe-local serve --host 0.0.0.0 --port 8000
```

### Continuous ingest pipeline

```bash
transcribe-local record start
transcribe-local vad start
transcribe-local upload start
```

`upload start` uses configured `server_url` by default.

### Browser capture (phone/any device)

Start server and job runner, then open:

```text
http://<server-ip>:8000/capture
```

The browser recorder streams chunks to the server and queues the final file for background transcription.

## Credentials

- `OPENAI_API_KEY`: required for `openai` and `hybrid` modes
- `HF_TOKEN`: required for `local` and `hybrid` diarization

Set credentials via `transcribe-local setup` or manually in `~/.transcribe_local/.env`.

## Troubleshooting

Run diagnostics:

```bash
transcribe-local doctor
```

Common failures:

- Missing `ffmpeg` in `PATH`
- Missing API keys/tokens
- Missing pyannote model access on Hugging Face
- Wrong uploader server URL
