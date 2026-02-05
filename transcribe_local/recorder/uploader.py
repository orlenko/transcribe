"""Upload daemon for sending VAD-approved recordings to remote server.

Monitors the ready folder and uploads files to the transcription web UI.
"""

import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import requests

from ..paths import get_data_dir

# Default settings
DEFAULT_READY_DIR = get_data_dir() / "ready"
DEFAULT_UPLOADED_DIR = get_data_dir() / "uploaded"
DEFAULT_CHECK_INTERVAL = 30  # Check every 30 seconds
DEFAULT_SERVER_URL = "http://localhost:8000"


class Uploader:
    """Uploads audio files to remote transcription server."""

    def __init__(
        self,
        server_url: str = DEFAULT_SERVER_URL,
        ready_dir: Optional[Path] = None,
        uploaded_dir: Optional[Path] = None,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        mode: str = "hybrid",
        language: Optional[str] = None,
        delete_after_upload: bool = False,
    ):
        self.server_url = server_url.rstrip("/")
        self.ready_dir = Path(ready_dir or DEFAULT_READY_DIR)
        self.uploaded_dir = Path(uploaded_dir or DEFAULT_UPLOADED_DIR)
        self.check_interval = check_interval
        self.mode = mode
        self.language = language
        self.delete_after_upload = delete_after_upload

        # Create directories
        self.ready_dir.mkdir(parents=True, exist_ok=True)
        self.uploaded_dir.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._pid_file = self.ready_dir / ".uploader.pid"

    def upload_file(self, audio_path: Path) -> bool:
        """Upload a single audio file to the server.

        Args:
            audio_path: Path to audio file.

        Returns:
            True if upload succeeded, False otherwise.
        """
        upload_url = f"{self.server_url}/api/upload"

        try:
            with open(audio_path, "rb") as f:
                files = {"file": (audio_path.name, f, "audio/mpeg")}
                data = {"mode": self.mode}
                if self.language:
                    data["language"] = self.language

                response = requests.post(
                    upload_url,
                    files=files,
                    data=data,
                    timeout=300,  # 5 min timeout for large files
                )

            if response.status_code == 200:
                result = response.json()
                return True
            else:
                print(f"Upload failed: {response.status_code} - {response.text}", file=sys.stderr)
                return False

        except requests.exceptions.ConnectionError:
            print(f"Connection error: Cannot reach {self.server_url}", file=sys.stderr)
            return False
        except requests.exceptions.Timeout:
            print(f"Timeout uploading {audio_path.name}", file=sys.stderr)
            return False
        except Exception as e:
            print(f"Error uploading {audio_path.name}: {e}", file=sys.stderr)
            return False

    def process_file(self, audio_path: Path) -> str:
        """Process a single file: upload and move/delete.

        Returns:
            "uploaded", "failed", or "skipped"
        """
        print(f"Uploading: {audio_path.name}...", end=" ", flush=True)

        if self.upload_file(audio_path):
            print("success")

            if self.delete_after_upload:
                audio_path.unlink()
            else:
                # Move to uploaded folder
                dest_path = self.uploaded_dir / audio_path.name
                audio_path.rename(dest_path)

            return "uploaded"
        else:
            print("failed (will retry)")
            return "failed"

    def process_folder(self) -> dict[str, int]:
        """Process all audio files in the ready folder.

        Returns:
            Dict with counts of uploaded, failed files.
        """
        results = {"uploaded": 0, "failed": 0, "skipped": 0}

        # Find audio files
        audio_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".aac", ".webm", ".opus"}
        audio_files = [
            f for f in self.ready_dir.iterdir()
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]

        for audio_path in sorted(audio_files):
            result = self.process_file(audio_path)
            results[result] += 1

        return results

    def check_server(self) -> bool:
        """Check if the server is reachable."""
        try:
            response = requests.get(f"{self.server_url}/api/jobs", timeout=10)
            return response.status_code == 200
        except Exception:
            return False

    def start(self) -> None:
        """Start continuous upload monitoring."""
        self._running = True

        # Write PID file
        self._pid_file.write_text(str(os.getpid()))

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        print(f"Uploader started (PID: {os.getpid()})")
        print(f"Server URL: {self.server_url}")
        print(f"Ready folder: {self.ready_dir}")
        print(f"Uploaded folder: {self.uploaded_dir}")
        print(f"Mode: {self.mode}")
        print(f"Check interval: {self.check_interval}s")
        print("Press Ctrl+C to stop")

        # Initial server check
        if self.check_server():
            print("Server connection: OK")
        else:
            print("Server connection: FAILED (will retry)")

        try:
            while self._running:
                results = self.process_folder()
                if results["uploaded"] or results["failed"]:
                    print(f"Processed: {results['uploaded']} uploaded, {results['failed']} failed")
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
        print("Uploader stopped")

    @classmethod
    def is_running(cls, ready_dir: Optional[Path] = None) -> Optional[int]:
        """Check if an uploader is running."""
        pid_file = Path(ready_dir or DEFAULT_READY_DIR) / ".uploader.pid"
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
    def stop_running(cls, ready_dir: Optional[Path] = None) -> bool:
        """Stop a running uploader."""
        pid = cls.is_running(ready_dir)
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

    parser = argparse.ArgumentParser(description="Upload recordings to transcription server")
    parser.add_argument("command", choices=["start", "stop", "status", "upload"])
    parser.add_argument("--server", "-s", default=DEFAULT_SERVER_URL, help="Server URL")
    parser.add_argument("--ready-dir", "-r", type=Path, default=DEFAULT_READY_DIR)
    parser.add_argument("--uploaded-dir", "-u", type=Path, default=DEFAULT_UPLOADED_DIR)
    parser.add_argument("--interval", "-i", type=int, default=DEFAULT_CHECK_INTERVAL)
    parser.add_argument("--mode", "-m", choices=["hybrid", "openai", "local"], default="hybrid")
    parser.add_argument("--language", "-l", help="Language code")
    parser.add_argument("--delete", action="store_true", help="Delete files after upload")

    args = parser.parse_args()

    if args.command == "status":
        pid = Uploader.is_running(args.ready_dir)
        if pid:
            print(f"Uploader is running (PID: {pid})")
        else:
            print("Uploader is not running")
    elif args.command == "stop":
        if Uploader.stop_running(args.ready_dir):
            print("Uploader stopped")
        else:
            print("Uploader is not running")
    elif args.command in ("start", "upload"):
        uploader = Uploader(
            server_url=args.server,
            ready_dir=args.ready_dir,
            uploaded_dir=args.uploaded_dir,
            check_interval=args.interval,
            mode=args.mode,
            language=args.language,
            delete_after_upload=args.delete,
        )

        if args.command == "upload":
            # One-time upload
            results = uploader.process_folder()
            print(f"Results: {results['uploaded']} uploaded, {results['failed']} failed")
        else:
            # Continuous monitoring
            uploader.start()


if __name__ == "__main__":
    main()
