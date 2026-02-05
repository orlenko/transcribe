"""Continuous audio recording daemon.

Records audio in fixed-duration chunks to a specified directory.
"""

import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..paths import get_data_dir

# Default settings
DEFAULT_CHUNK_DURATION = 600  # 10 minutes in seconds
DEFAULT_OUTPUT_DIR = get_data_dir() / "recordings"
DEFAULT_AUDIO_FORMAT = "mp3"
DEFAULT_BITRATE = "64k"  # Lower bitrate for speech (saves space)
DEFAULT_SAMPLE_RATE = 16000  # 16kHz is good for speech


class RecordingDaemon:
    """Continuous audio recording daemon."""

    def __init__(
        self,
        output_dir: Optional[Path] = None,
        chunk_duration: int = DEFAULT_CHUNK_DURATION,
        audio_format: str = DEFAULT_AUDIO_FORMAT,
        bitrate: str = DEFAULT_BITRATE,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        input_device: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir or DEFAULT_OUTPUT_DIR)
        self.chunk_duration = chunk_duration
        self.audio_format = audio_format
        self.bitrate = bitrate
        self.sample_rate = sample_rate
        self.input_device = input_device

        self._running = False
        self._current_process: Optional[subprocess.Popen] = None
        self._pid_file = self.output_dir / ".recorder.pid"

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_input_device(self) -> str:
        """Get the audio input device string for FFmpeg."""
        if sys.platform == "darwin":
            # macOS: use avfoundation
            # Device index 0 is usually the default mic
            # Use ":0" for default audio input (no video)
            return self.input_device or ":0"
        elif sys.platform == "linux":
            # Linux: use pulse or alsa
            return self.input_device or "default"
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

    def _get_ffmpeg_command(self, output_path: Path) -> list[str]:
        """Build FFmpeg command for recording."""
        if sys.platform == "darwin":
            # macOS
            cmd = [
                "ffmpeg",
                "-f", "avfoundation",
                "-i", self._get_input_device(),
                "-t", str(self.chunk_duration),
                "-ar", str(self.sample_rate),
                "-ac", "1",  # Mono
                "-c:a", "libmp3lame" if self.audio_format == "mp3" else "aac",
                "-b:a", self.bitrate,
                "-y",  # Overwrite
                str(output_path),
            ]
        elif sys.platform == "linux":
            # Linux with PulseAudio
            cmd = [
                "ffmpeg",
                "-f", "pulse",
                "-i", self._get_input_device(),
                "-t", str(self.chunk_duration),
                "-ar", str(self.sample_rate),
                "-ac", "1",
                "-c:a", "libmp3lame" if self.audio_format == "mp3" else "aac",
                "-b:a", self.bitrate,
                "-y",
                str(output_path),
            ]
        else:
            raise RuntimeError(f"Unsupported platform: {sys.platform}")

        return cmd

    def _generate_filename(self) -> str:
        """Generate filename with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"recording-{timestamp}.{self.audio_format}"

    def record_chunk(self) -> Optional[Path]:
        """Record a single chunk of audio.

        Returns:
            Path to the recorded file, or None if recording failed.
        """
        output_path = self.output_dir / self._generate_filename()
        cmd = self._get_ffmpeg_command(output_path)

        try:
            # Run FFmpeg with suppressed output
            self._current_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            self._current_process.wait()

            if output_path.exists() and output_path.stat().st_size > 0:
                return output_path
            return None

        except Exception as e:
            print(f"Recording error: {e}", file=sys.stderr)
            return None
        finally:
            self._current_process = None

    def start(self) -> None:
        """Start continuous recording."""
        self._running = True

        # Write PID file
        self._pid_file.write_text(str(os.getpid()))

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        print(f"Recording daemon started (PID: {os.getpid()})")
        print(f"Output directory: {self.output_dir}")
        print(f"Chunk duration: {self.chunk_duration}s")
        print("Press Ctrl+C to stop")

        try:
            while self._running:
                chunk_path = self.record_chunk()
                if chunk_path:
                    print(f"Recorded: {chunk_path.name}")
                else:
                    print("Recording failed, retrying in 5s...")
                    time.sleep(5)
        finally:
            self._cleanup()

    def stop(self) -> None:
        """Stop recording."""
        self._running = False
        if self._current_process:
            self._current_process.terminate()
            self._current_process.wait()

    def _handle_signal(self, signum, frame) -> None:
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, stopping...")
        self.stop()

    def _cleanup(self) -> None:
        """Cleanup on shutdown."""
        if self._pid_file.exists():
            self._pid_file.unlink()
        print("Recording daemon stopped")

    @classmethod
    def is_running(cls, output_dir: Optional[Path] = None) -> Optional[int]:
        """Check if a recording daemon is running.

        Returns:
            PID if running, None otherwise.
        """
        pid_file = Path(output_dir or DEFAULT_OUTPUT_DIR) / ".recorder.pid"
        if not pid_file.exists():
            return None

        try:
            pid = int(pid_file.read_text().strip())
            # Check if process exists
            os.kill(pid, 0)
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            # Stale PID file
            pid_file.unlink(missing_ok=True)
            return None

    @classmethod
    def stop_running(cls, output_dir: Optional[Path] = None) -> bool:
        """Stop a running recording daemon.

        Returns:
            True if daemon was stopped, False if not running.
        """
        pid = cls.is_running(output_dir)
        if pid is None:
            return False

        try:
            os.kill(pid, signal.SIGTERM)
            # Wait for process to exit
            for _ in range(10):
                time.sleep(0.5)
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return True
            # Force kill if still running
            os.kill(pid, signal.SIGKILL)
            return True
        except ProcessLookupError:
            return True
        except PermissionError:
            print(f"Permission denied to stop PID {pid}", file=sys.stderr)
            return False


def list_audio_devices() -> None:
    """List available audio input devices."""
    if sys.platform == "darwin":
        print("Available audio devices (macOS):")
        subprocess.run(
            ["ffmpeg", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
            stderr=subprocess.STDOUT,
        )
    elif sys.platform == "linux":
        print("Available audio devices (Linux/PulseAudio):")
        subprocess.run(["pactl", "list", "sources", "short"])
    else:
        print(f"Unsupported platform: {sys.platform}")


def main():
    """CLI entry point for testing."""
    import argparse

    parser = argparse.ArgumentParser(description="Continuous audio recording daemon")
    parser.add_argument("command", choices=["start", "stop", "status", "devices"])
    parser.add_argument("--output-dir", "-o", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--duration", "-d", type=int, default=DEFAULT_CHUNK_DURATION)
    parser.add_argument("--device", help="Input device (platform-specific)")

    args = parser.parse_args()

    if args.command == "devices":
        list_audio_devices()
    elif args.command == "status":
        pid = RecordingDaemon.is_running(args.output_dir)
        if pid:
            print(f"Recording daemon is running (PID: {pid})")
        else:
            print("Recording daemon is not running")
    elif args.command == "stop":
        if RecordingDaemon.stop_running(args.output_dir):
            print("Recording daemon stopped")
        else:
            print("Recording daemon is not running")
    elif args.command == "start":
        daemon = RecordingDaemon(
            output_dir=args.output_dir,
            chunk_duration=args.duration,
            input_device=args.device,
        )
        daemon.start()


if __name__ == "__main__":
    main()
