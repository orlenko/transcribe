"""Voice Activity Detection filter.

Monitors a folder for audio files and deletes those without speech.
Files with speech are moved to a "ready" folder for transcription.
"""

import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from ..paths import get_data_dir

# Default settings
DEFAULT_RECORDINGS_DIR = get_data_dir() / "recordings"
DEFAULT_READY_DIR = get_data_dir() / "ready"
DEFAULT_MIN_SPEECH_RATIO = 0.05  # At least 5% speech to keep
DEFAULT_CHECK_INTERVAL = 30  # Check every 30 seconds


class SileroVAD:
    """Silero VAD wrapper for speech detection."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None
        self._get_speech_timestamps = None

    @property
    def model(self):
        """Lazy-load Silero VAD model."""
        if self._model is None:
            # Load Silero VAD from torch hub
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
            )
            self._model = model
            self._get_speech_timestamps = utils[0]  # get_speech_timestamps function
        return self._model

    def get_speech_ratio(self, audio_path: str, sample_rate: int = 16000) -> float:
        """Calculate the ratio of speech in an audio file.

        Args:
            audio_path: Path to audio file.
            sample_rate: Expected sample rate (will resample if different).

        Returns:
            Ratio of speech (0.0 to 1.0).
        """
        import torchaudio

        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_path)

            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Resample if needed
            if sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sr, sample_rate)
                waveform = resampler(waveform)

            # Flatten to 1D
            waveform = waveform.squeeze()

            # Get speech timestamps
            speech_timestamps = self._get_speech_timestamps(
                waveform,
                self.model,
                sampling_rate=sample_rate,
                threshold=0.5,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100,
            )

            if not speech_timestamps:
                return 0.0

            # Calculate total speech duration
            total_samples = len(waveform)
            speech_samples = sum(
                ts['end'] - ts['start']
                for ts in speech_timestamps
            )

            return speech_samples / total_samples

        except Exception as e:
            print(f"Error analyzing {audio_path}: {e}", file=sys.stderr)
            return -1.0  # Return -1 to indicate error


class WebRTCVAD:
    """WebRTC VAD wrapper (faster, less accurate than Silero)."""

    def __init__(self, aggressiveness: int = 2):
        """Initialize WebRTC VAD.

        Args:
            aggressiveness: 0-3, higher = more aggressive filtering.
        """
        try:
            import webrtcvad
            self.vad = webrtcvad.Vad(aggressiveness)
        except ImportError:
            raise ImportError("webrtcvad not installed. Run: pip install webrtcvad")

    def get_speech_ratio(self, audio_path: str, sample_rate: int = 16000) -> float:
        """Calculate the ratio of speech in an audio file."""
        import wave
        import tempfile
        import subprocess

        try:
            # Convert to WAV (webrtcvad needs raw PCM)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name

            subprocess.run([
                "ffmpeg", "-y", "-i", audio_path,
                "-ar", str(sample_rate),
                "-ac", "1",
                "-f", "wav",
                tmp_path,
            ], capture_output=True, check=True)

            # Read WAV file
            with wave.open(tmp_path, 'rb') as wf:
                sample_rate = wf.getframerate()
                num_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                audio_data = wf.readframes(wf.getnframes())

            os.unlink(tmp_path)

            # WebRTC VAD needs 16-bit PCM at 8000, 16000, 32000, or 48000 Hz
            if sample_rate not in (8000, 16000, 32000, 48000):
                print(f"Unsupported sample rate: {sample_rate}", file=sys.stderr)
                return -1.0

            # Process in 30ms frames
            frame_duration_ms = 30
            frame_size = int(sample_rate * frame_duration_ms / 1000) * sample_width
            num_frames = len(audio_data) // frame_size

            if num_frames == 0:
                return 0.0

            speech_frames = 0
            for i in range(num_frames):
                frame = audio_data[i * frame_size:(i + 1) * frame_size]
                if len(frame) == frame_size:
                    if self.vad.is_speech(frame, sample_rate):
                        speech_frames += 1

            return speech_frames / num_frames

        except Exception as e:
            print(f"Error analyzing {audio_path}: {e}", file=sys.stderr)
            return -1.0


class VADFilter:
    """Monitors folder and filters audio files based on speech content."""

    def __init__(
        self,
        recordings_dir: Optional[Path] = None,
        ready_dir: Optional[Path] = None,
        min_speech_ratio: float = DEFAULT_MIN_SPEECH_RATIO,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        vad_engine: str = "silero",  # "silero" or "webrtc"
        delete_silent: bool = True,
    ):
        self.recordings_dir = Path(recordings_dir or DEFAULT_RECORDINGS_DIR)
        self.ready_dir = Path(ready_dir or DEFAULT_READY_DIR)
        self.min_speech_ratio = min_speech_ratio
        self.check_interval = check_interval
        self.delete_silent = delete_silent

        # Create directories
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.ready_dir.mkdir(parents=True, exist_ok=True)

        # Initialize VAD
        if vad_engine == "webrtc":
            self.vad = WebRTCVAD()
        else:
            self.vad = SileroVAD()

        self._running = False
        self._processed_files: set[str] = set()
        self._pid_file = self.recordings_dir / ".vad_filter.pid"

    def _is_file_complete(self, audio_path: Path) -> bool:
        """Check if a file is complete (not still being written).

        Uses multiple checks:
        1. File must not have been modified in last 60 seconds
        2. File size must be stable (same size after 2 seconds)
        """
        try:
            # Check 1: Modified time
            if time.time() - audio_path.stat().st_mtime < 60:
                return False

            # Check 2: File size stability
            size1 = audio_path.stat().st_size
            time.sleep(2)
            if not audio_path.exists():
                return False
            size2 = audio_path.stat().st_size

            return size1 == size2 and size1 > 0
        except (OSError, FileNotFoundError):
            return False

    def process_file(self, audio_path: Path) -> str:
        """Process a single audio file.

        Returns:
            "kept", "deleted", "error", or "skipped"
        """
        if audio_path.name in self._processed_files:
            return "skipped"

        # Skip files that are still being written
        if not self._is_file_complete(audio_path):
            return "skipped"

        print(f"Analyzing: {audio_path.name}...", end=" ", flush=True)

        speech_ratio = self.vad.get_speech_ratio(str(audio_path))

        if speech_ratio < 0:
            print("error (will retry)")
            # Don't add to processed_files so we retry later
            return "error"

        print(f"{speech_ratio:.1%} speech", end=" ")

        if speech_ratio >= self.min_speech_ratio:
            # Move to ready folder
            dest_path = self.ready_dir / audio_path.name
            shutil.move(str(audio_path), str(dest_path))
            print("-> kept")
            self._processed_files.add(audio_path.name)
            return "kept"
        else:
            # Delete or skip
            if self.delete_silent:
                audio_path.unlink()
                print("-> deleted")
            else:
                print("-> skipped (below threshold)")
            self._processed_files.add(audio_path.name)
            return "deleted"

    def process_folder(self) -> dict[str, int]:
        """Process all audio files in the recordings folder.

        Returns:
            Dict with counts of kept, deleted, error, skipped files.
        """
        results = {"kept": 0, "deleted": 0, "error": 0, "skipped": 0}

        # Find audio files
        audio_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".aac", ".webm", ".opus"}
        audio_files = [
            f for f in self.recordings_dir.iterdir()
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]

        for audio_path in sorted(audio_files):
            result = self.process_file(audio_path)
            results[result] += 1

        return results

    def start(self) -> None:
        """Start continuous monitoring."""
        self._running = True

        # Write PID file
        self._pid_file.write_text(str(os.getpid()))

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        print(f"VAD filter started (PID: {os.getpid()})")
        print(f"Monitoring: {self.recordings_dir}")
        print(f"Ready folder: {self.ready_dir}")
        print(f"Min speech ratio: {self.min_speech_ratio:.0%}")
        print(f"Check interval: {self.check_interval}s")
        print("Press Ctrl+C to stop")

        try:
            while self._running:
                results = self.process_folder()
                if results["kept"] or results["deleted"]:
                    print(f"Processed: {results['kept']} kept, {results['deleted']} deleted")
                time.sleep(self.check_interval)
        finally:
            self._cleanup()

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False

    def _handle_signal(self, signum, frame) -> None:
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, stopping...")
        self.stop()

    def _cleanup(self) -> None:
        """Cleanup on shutdown."""
        if self._pid_file.exists():
            self._pid_file.unlink()
        print("VAD filter stopped")

    @classmethod
    def is_running(cls, recordings_dir: Optional[Path] = None) -> Optional[int]:
        """Check if a VAD filter is running."""
        pid_file = Path(recordings_dir or DEFAULT_RECORDINGS_DIR) / ".vad_filter.pid"
        if not pid_file.exists():
            return None

        try:
            pid = int(pid_file.read_text().strip())
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            pid_file.unlink(missing_ok=True)
            return None

    @classmethod
    def stop_running(cls, recordings_dir: Optional[Path] = None) -> bool:
        """Stop a running VAD filter."""
        pid = cls.is_running(recordings_dir)
        if pid is None:
            return False

        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(10):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True
            os.kill(pid, signal.SIGKILL)
            return True
        except ProcessLookupError:
            return True
        except PermissionError:
            print(f"Permission denied to stop PID {pid}", file=sys.stderr)
            return False


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="VAD filter for audio recordings")
    parser.add_argument("command", choices=["start", "stop", "status", "process"])
    parser.add_argument("--recordings-dir", "-r", type=Path, default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--ready-dir", "-o", type=Path, default=DEFAULT_READY_DIR)
    parser.add_argument("--min-speech", "-m", type=float, default=DEFAULT_MIN_SPEECH_RATIO)
    parser.add_argument("--interval", "-i", type=int, default=DEFAULT_CHECK_INTERVAL)
    parser.add_argument("--engine", choices=["silero", "webrtc"], default="silero")
    parser.add_argument("--no-delete", action="store_true", help="Don't delete silent files")

    args = parser.parse_args()

    if args.command == "status":
        pid = VADFilter.is_running(args.recordings_dir)
        if pid:
            print(f"VAD filter is running (PID: {pid})")
        else:
            print("VAD filter is not running")
    elif args.command == "stop":
        if VADFilter.stop_running(args.recordings_dir):
            print("VAD filter stopped")
        else:
            print("VAD filter is not running")
    elif args.command in ("start", "process"):
        vad_filter = VADFilter(
            recordings_dir=args.recordings_dir,
            ready_dir=args.ready_dir,
            min_speech_ratio=args.min_speech,
            check_interval=args.interval,
            vad_engine=args.engine,
            delete_silent=not args.no_delete,
        )

        if args.command == "process":
            # One-time processing
            results = vad_filter.process_folder()
            print(f"Results: {results['kept']} kept, {results['deleted']} deleted, {results['error']} errors")
        else:
            # Continuous monitoring
            vad_filter.start()


if __name__ == "__main__":
    main()
