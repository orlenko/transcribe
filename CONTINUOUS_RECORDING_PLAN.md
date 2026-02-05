# Continuous Recording & Transcription Plan

## Overview

Create a background process on macOS that continuously records audio, filters for speech, and automatically queues recordings for transcription. This creates an "always-on" transcription service for reviewing past conversations.

## Architecture

```
┌─────────────────┐     ┌───────────────┐     ┌─────────────┐
│  Recorder       │────▶│  VAD Filter   │────▶│  Web UI     │
│  (launchd)      │     │  (discard     │     │  Auto-queue │
│  10-min chunks  │     │   silence)    │     │             │
└─────────────────┘     └───────────────┘     └─────────────┘
         │                                           │
         └──── ~/Recordings/chunks/ ─────────────────┘
```

## Recording Options

| Method | Pros | Cons |
|--------|------|------|
| **FFmpeg** | Simple, reliable, already common | Needs mic permission |
| **SoX** | Lightweight, good for scripts | Less common on Mac |
| **PyAudio** | Python native, easy integration | Can be finicky to install |
| **BlackHole + FFmpeg** | Can capture system audio too | Extra setup |

**Recommended**: FFmpeg for simplicity and reliability.

## Voice Activity Detection (VAD)

Options for filtering out silent/non-speech chunks:

1. **WebRTC VAD** - Super fast (~1ms per chunk), lightweight, Python bindings (py-webrtcvad)
2. **Silero VAD** - ML-based, more accurate, still fast, PyTorch
3. **PyAnnote VAD** - Already installed with our project, excellent accuracy

**Recommended**: WebRTC VAD for speed, or Silero for accuracy.

## Implementation Plan

### 1. Recording Daemon

```python
# transcribe_local/recorder.py
- Record audio in 10-minute chunks
- Save to configurable watch folder
- Run as launchd service (macOS) or systemd (Linux)
- Handle graceful shutdown
```

### 2. VAD Filter Script

```python
# transcribe_local/vad_filter.py
- Monitor watch folder for new chunks
- Run VAD on each chunk
- Calculate speech ratio (% of audio containing speech)
- Discard chunks below threshold (e.g., <10% speech)
- Move speech chunks to transcription queue
```

### 3. CLI Commands

```bash
transcribe-local record start    # Start recording daemon
transcribe-local record stop     # Stop recording daemon
transcribe-local record status   # Show daemon status
transcribe-local record config   # Configure settings
```

### 4. Web UI Integration

- Add "Watch Folder" feature to web UI
- Auto-queue new files for transcription
- Show recording status indicator
- Configure retention policy

### 5. Configuration

```json
{
  "recording": {
    "enabled": false,
    "input_device": "default",
    "chunk_duration_minutes": 10,
    "output_dir": "~/Recordings/chunks",
    "audio_format": "mp3",
    "audio_quality": "128k"
  },
  "vad": {
    "enabled": true,
    "min_speech_ratio": 0.1,
    "engine": "webrtc"
  },
  "retention": {
    "keep_days": 30,
    "auto_delete": true
  }
}
```

## Key Considerations

### Permissions
- macOS requires explicit microphone access permission
- First run will trigger system permission dialog
- Can be pre-authorized via MDM for enterprise deployment

### Privacy & Legal
- **Important**: Recording conversations has legal implications
- Many jurisdictions require consent from all parties (two-party consent)
- Consider adding:
  - Visual indicator when recording is active
  - Audio notification at start of recording
  - Easy pause/stop controls
  - Clear data retention policies

### Storage
- ~10MB per 10-minute chunk at 128kbps MP3
- ~1.4GB per day of continuous recording
- ~10GB per week
- Auto-cleanup of old recordings recommended

### Battery
- Continuous recording uses power
- Better suited for plugged-in scenarios
- Consider "record only when charging" option

### Recording Indicator
- macOS shows orange dot in menu bar when mic is in use
- Consider adding menu bar app for easy control

## Quick Feasibility Test

```bash
# List available audio devices
ffmpeg -f avfoundation -list_devices true -i ""

# Test 10-second recording from default mic
ffmpeg -f avfoundation -i ":0" -t 10 -c:a libmp3lame -b:a 128k test.mp3

# Play back
afplay test.mp3
```

## Dependencies to Add

```
py-webrtcvad>=2.0.10  # Fast VAD
# or
silero-vad            # Accurate VAD (uses torch, already installed)
```

## File Structure

```
transcribe_local/
├── recorder/
│   ├── __init__.py
│   ├── daemon.py        # Recording daemon
│   ├── vad.py           # Voice activity detection
│   ├── watcher.py       # Watch folder monitor
│   └── launchd.py       # macOS service management
└── ...
```

## Future Enhancements

1. **System audio capture** - Record Zoom/Meet calls (requires BlackHole or similar)
2. **Smart chunking** - Split on silence instead of fixed intervals
3. **Real-time transcription** - Stream to Whisper as recording happens
4. **Speaker-triggered recording** - Only record when specific voices detected
5. **Keyword detection** - Flag recordings containing specific words
6. **Mobile app** - Remote access to recordings and transcripts
