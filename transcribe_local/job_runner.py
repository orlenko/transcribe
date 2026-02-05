"""Standalone job runner daemon for processing uploaded audio files.

Monitors the uploads folder and processes files independently of the web app.
"""

import os
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

from .database import Database
from .env import load_transcribe_env
from .models import DiarizedSegment
from .timezone_utils import extract_recording_time
from .title_generator import format_transcript_for_title, generate_title
from .web.config import Settings, get_settings

# Default settings
DEFAULT_CHECK_INTERVAL = 30  # seconds


class JobRunner:
    """Processes audio files from the uploads folder."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        check_interval: int = DEFAULT_CHECK_INTERVAL,
        mode: Optional[str] = None,
        language: Optional[str] = None,
    ):
        self.settings = settings or get_settings()
        self.check_interval = check_interval
        self.mode = mode or self.settings.transcription_mode
        self.language = language or self.settings.default_language

        self.db = Database(self.settings.db_path)
        self.uploads_dir = self.settings.upload_dir
        self.processed_dir = self.settings.data_dir / "processed"

        # Create directories
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        self._running = False
        self._pid_file = self.settings.data_dir / ".job_runner.pid"
        self._current_file: Optional[str] = None

    def _get_unprocessed_files(self) -> list[Path]:
        """Get list of audio files that haven't been processed yet."""
        audio_extensions = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac", ".aac", ".webm", ".opus"}

        audio_files = [
            f for f in self.uploads_dir.iterdir()
            if f.is_file() and f.suffix.lower() in audio_extensions
        ]

        # Filter out files that are already in the database
        unprocessed = []
        for audio_path in audio_files:
            # Check if file hash already exists in database
            file_hash = self.db._compute_file_hash(str(audio_path))
            existing = self.db.get_audio_file_by_hash(file_hash)
            if not existing:
                unprocessed.append(audio_path)

        return sorted(unprocessed, key=lambda f: f.stat().st_mtime)

    def _is_file_complete(self, audio_path: Path) -> bool:
        """Check if file is complete (not still being written)."""
        try:
            # File must not have been modified in last 10 seconds
            if time.time() - audio_path.stat().st_mtime < 10:
                return False

            # File size must be stable
            size1 = audio_path.stat().st_size
            time.sleep(2)
            if not audio_path.exists():
                return False
            size2 = audio_path.stat().st_size

            return size1 == size2 and size1 > 0
        except (OSError, FileNotFoundError):
            return False

    def process_file(self, audio_path: Path) -> bool:
        """Process a single audio file.

        Returns:
            True if successful, False otherwise.
        """
        self._current_file = audio_path.name
        print(f"\n{'='*60}")
        print(f"Processing: {audio_path.name}")
        print(f"Mode: {self.mode}")
        print(f"{'='*60}")

        try:
            if self.mode == "openai":
                self._process_openai_only(audio_path)
            elif self.mode == "local":
                self._process_local(audio_path)
            else:  # hybrid
                self._process_hybrid(audio_path)

            # Move to processed folder
            dest_path = self.processed_dir / audio_path.name
            if dest_path.exists():
                # Add timestamp to avoid collision
                stem = audio_path.stem
                suffix = audio_path.suffix
                timestamp = int(time.time())
                dest_path = self.processed_dir / f"{stem}_{timestamp}{suffix}"

            audio_path.rename(dest_path)
            print(f"Moved to: {dest_path}")
            print(f"SUCCESS: {audio_path.name}")
            return True

        except Exception as e:
            print(f"ERROR processing {audio_path.name}: {e}", file=sys.stderr)
            traceback.print_exc()
            return False
        finally:
            self._current_file = None

    def _process_openai_only(self, audio_path: Path) -> None:
        """Process with OpenAI only (no diarization)."""
        import openai

        load_transcribe_env()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        openai.api_key = api_key

        print("Transcribing with OpenAI...")

        file_size = audio_path.stat().st_size
        max_size = 24 * 1024 * 1024

        if file_size > max_size:
            segments, detected_lang = self._transcribe_openai_chunked(audio_path)
        else:
            with open(audio_path, "rb") as f:
                result = openai.audio.transcriptions.create(
                    model=self.settings.openai_model,
                    file=f,
                    language=self.language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            detected_lang = result.language if hasattr(result, "language") else self.language or "en"
            segments = []
            if hasattr(result, "segments") and result.segments:
                for seg in result.segments:
                    segments.append({
                        "start": seg.start if hasattr(seg, "start") else seg["start"],
                        "end": seg.end if hasattr(seg, "end") else seg["end"],
                        "text": seg.text if hasattr(seg, "text") else seg["text"],
                    })

        print(f"Got {len(segments)} segments, language: {detected_lang}")
        print("Saving to database...")

        recorded_at = extract_recording_time(str(audio_path))
        audio_record = self.db.add_audio_file(str(audio_path), recorded_at=recorded_at)
        transcript_record = self.db.add_transcript(
            audio_id=audio_record.id,
            model_name=f"openai/{self.settings.openai_model}",
            language=detected_lang,
        )

        for idx, seg in enumerate(segments):
            self.db.add_segment(
                transcript_id=transcript_record.id,
                segment_index=idx,
                start_time=seg["start"],
                end_time=seg["end"],
                text=seg["text"].strip(),
                speaker_label="SPEAKER_00",
            )

        # Generate title
        print("Generating title...")
        transcript_text = " ".join(seg["text"].strip() for seg in segments)
        title = generate_title(transcript_text)
        self.db.update_transcript_title(transcript_record.id, title)
        print(f"Title: {title}")

        print(f"Saved transcript ID: {transcript_record.id}")

    def _process_hybrid(self, audio_path: Path) -> None:
        """Process with OpenAI transcription + local diarization."""
        import openai

        from .diarizer import Diarizer
        from .speaker_matcher import SpeakerMatcher, update_segments_with_matches

        load_transcribe_env()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        openai.api_key = api_key

        # Step 1: Transcribe with OpenAI
        print("Step 1/4: Transcribing with OpenAI...")

        file_size = audio_path.stat().st_size
        max_size = 24 * 1024 * 1024

        if file_size > max_size:
            openai_segments, detected_lang = self._transcribe_openai_chunked(audio_path)
        else:
            with open(audio_path, "rb") as f:
                result = openai.audio.transcriptions.create(
                    model=self.settings.openai_model,
                    file=f,
                    language=self.language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"],
                )

            detected_lang = result.language if hasattr(result, "language") else self.language or "en"
            openai_segments = []
            if hasattr(result, "segments") and result.segments:
                for seg in result.segments:
                    openai_segments.append({
                        "start": seg.start if hasattr(seg, "start") else seg["start"],
                        "end": seg.end if hasattr(seg, "end") else seg["end"],
                        "text": seg.text if hasattr(seg, "text") else seg["text"],
                    })

        print(f"Got {len(openai_segments)} segments, language: {detected_lang}")

        # Step 2: Diarize locally
        print("Step 2/4: Performing speaker diarization...")

        diarizer = Diarizer()
        diarize_result = diarizer.diarize(
            str(audio_path),
            min_speakers=self.settings.min_speakers,
            max_speakers=self.settings.max_speakers,
        )

        # Step 3: Merge
        print("Step 3/4: Merging transcription with speakers...")

        segments = self._merge_transcription_with_diarization(openai_segments, diarize_result)
        unique_speakers = diarizer.get_unique_speakers(segments)
        print(f"Detected {len(unique_speakers)} speakers: {unique_speakers}")

        # Step 4: Match speakers
        print("Step 4/4: Matching known speakers...")

        matcher = SpeakerMatcher(self.db, threshold=self.settings.speaker_match_threshold)
        matches = []

        speaker_embeddings = {}
        for speaker_label in unique_speakers:
            embeddings = diarizer.extract_embeddings_for_speaker(
                str(audio_path), segments, speaker_label, max_embeddings=1
            )
            if embeddings:
                speaker_embeddings[speaker_label] = embeddings[0][0]

        if speaker_embeddings:
            matches = matcher.match_speakers(speaker_embeddings)
            segments = update_segments_with_matches(segments, matches)

            for match in matches:
                if match.matched_speaker:
                    print(f"  {match.speaker_label} -> {match.matched_speaker.name} ({match.confidence:.2f})")
                else:
                    print(f"  {match.speaker_label} -> Unknown")

        # Save to database
        print("Saving to database...")

        recorded_at = extract_recording_time(str(audio_path))
        audio_record = self.db.add_audio_file(str(audio_path), recorded_at=recorded_at)
        transcript_record = self.db.add_transcript(
            audio_id=audio_record.id,
            model_name=f"openai/{self.settings.openai_model}",
            language=detected_lang,
        )

        for idx, seg in enumerate(segments):
            speaker_id = None
            confidence = None
            for match in matches:
                if match.matched_speaker and match.speaker_label in seg.speaker_label:
                    speaker_id = match.matched_speaker.id
                    confidence = match.confidence
                    break

            self.db.add_segment(
                transcript_id=transcript_record.id,
                segment_index=idx,
                start_time=seg.start,
                end_time=seg.end,
                text=seg.text,
                speaker_label=seg.speaker_label,
                speaker_id=speaker_id,
                confidence=confidence,
            )

        # Generate title
        print("Generating title...")
        transcript_text = format_transcript_for_title(segments)
        title = generate_title(transcript_text)
        self.db.update_transcript_title(transcript_record.id, title)
        print(f"Title: {title}")

        print(f"Saved transcript ID: {transcript_record.id}")

    def _process_local(self, audio_path: Path) -> None:
        """Process with fully local transcription."""
        from .diarizer import Diarizer
        from .speaker_matcher import SpeakerMatcher, update_segments_with_matches
        from .transcriber import Transcriber

        # Step 1: Transcribe locally
        print("Step 1/4: Transcribing locally...")

        transcriber = Transcriber(model_name=self.settings.local_model)
        diarizer = Diarizer()

        aligned_result, detected_lang = transcriber.transcribe_and_align(
            str(audio_path),
            language=self.language,
        )

        print(f"Language: {detected_lang}")

        # Step 2: Diarize
        print("Step 2/4: Performing speaker diarization...")

        diarize_result = diarizer.diarize(
            str(audio_path),
            min_speakers=self.settings.min_speakers,
            max_speakers=self.settings.max_speakers,
        )

        # Step 3: Assign speakers
        print("Step 3/4: Assigning speakers...")

        segments = diarizer.assign_speakers(diarize_result, aligned_result)
        unique_speakers = diarizer.get_unique_speakers(segments)
        print(f"Detected {len(unique_speakers)} speakers: {unique_speakers}")

        # Step 4: Match speakers
        print("Step 4/4: Matching known speakers...")

        matcher = SpeakerMatcher(self.db, threshold=self.settings.speaker_match_threshold)
        matches = []

        speaker_embeddings = {}
        for speaker_label in unique_speakers:
            embeddings = diarizer.extract_embeddings_for_speaker(
                str(audio_path), segments, speaker_label, max_embeddings=1
            )
            if embeddings:
                speaker_embeddings[speaker_label] = embeddings[0][0]

        if speaker_embeddings:
            matches = matcher.match_speakers(speaker_embeddings)
            segments = update_segments_with_matches(segments, matches)

        # Save to database
        print("Saving to database...")

        recorded_at = extract_recording_time(str(audio_path))
        audio_record = self.db.add_audio_file(str(audio_path), recorded_at=recorded_at)
        transcript_record = self.db.add_transcript(
            audio_id=audio_record.id,
            model_name=self.settings.local_model,
            language=detected_lang,
        )

        for idx, seg in enumerate(segments):
            speaker_id = None
            confidence = None
            for match in matches:
                if match.matched_speaker and match.speaker_label in seg.speaker_label:
                    speaker_id = match.matched_speaker.id
                    confidence = match.confidence
                    break

            self.db.add_segment(
                transcript_id=transcript_record.id,
                segment_index=idx,
                start_time=seg.start,
                end_time=seg.end,
                text=seg.text,
                speaker_label=seg.speaker_label,
                speaker_id=speaker_id,
                confidence=confidence,
            )

        # Generate title
        print("Generating title...")
        transcript_text = format_transcript_for_title(segments)
        title = generate_title(transcript_text)
        self.db.update_transcript_title(transcript_record.id, title)
        print(f"Title: {title}")

        print(f"Saved transcript ID: {transcript_record.id}")

    def _transcribe_openai_chunked(self, audio_path: Path) -> tuple[list[dict], str]:
        """Transcribe large audio file in chunks."""
        import tempfile

        import openai
        from pydub import AudioSegment

        print("Loading audio for chunking...")

        audio = AudioSegment.from_file(str(audio_path))
        total_duration_ms = len(audio)
        chunk_duration_ms = 10 * 60 * 1000  # 10 minutes

        chunks = []
        start_ms = 0
        while start_ms < total_duration_ms:
            end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
            chunks.append((start_ms, end_ms))
            start_ms = end_ms

        print(f"Split into {len(chunks)} chunks")

        all_segments = []
        detected_lang = self.language or "en"

        for i, (start_ms, end_ms) in enumerate(chunks):
            chunk_start_sec = start_ms / 1000
            print(f"  Transcribing chunk {i + 1}/{len(chunks)}...")

            chunk_audio = audio[start_ms:end_ms]

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
                chunk_audio.export(tmp_path, format="mp3")

            try:
                with open(tmp_path, "rb") as f:
                    result = openai.audio.transcriptions.create(
                        model=self.settings.openai_model,
                        file=f,
                        language=self.language,
                        response_format="verbose_json",
                        timestamp_granularities=["segment", "word"],
                    )

                if i == 0 and hasattr(result, "language") and result.language:
                    detected_lang = result.language

                if hasattr(result, "segments") and result.segments:
                    for seg in result.segments:
                        seg_start = seg.start if hasattr(seg, "start") else seg["start"]
                        seg_end = seg.end if hasattr(seg, "end") else seg["end"]
                        seg_text = seg.text if hasattr(seg, "text") else seg["text"]

                        all_segments.append({
                            "start": seg_start + chunk_start_sec,
                            "end": seg_end + chunk_start_sec,
                            "text": seg_text,
                        })
            finally:
                os.unlink(tmp_path)

        return all_segments, detected_lang

    def _merge_transcription_with_diarization(
        self,
        openai_segments: list[dict],
        diarize_result,
    ) -> list[DiarizedSegment]:
        """Merge OpenAI transcription with pyannote diarization."""
        segments = []

        for seg in openai_segments:
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_text = seg["text"].strip()

            if not seg_text:
                continue

            speaker = self._find_speaker_for_segment(seg_start, seg_end, diarize_result)

            segments.append(
                DiarizedSegment(
                    start=seg_start,
                    end=seg_end,
                    text=seg_text,
                    speaker_label=speaker,
                )
            )

        return segments

    def _find_speaker_for_segment(self, start: float, end: float, diarize_result) -> str:
        """Find dominant speaker for a segment."""
        if diarize_result is None or len(diarize_result) == 0:
            return "UNKNOWN"

        speaker_overlap = {}

        for _, row in diarize_result.iterrows():
            overlap_start = max(start, row["start"])
            overlap_end = min(end, row["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                speaker = row["speaker"]
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + overlap

        if not speaker_overlap:
            return "UNKNOWN"

        return max(speaker_overlap, key=speaker_overlap.get)

    def process_backlog(self) -> dict[str, int]:
        """Process all unprocessed files once.

        Returns:
            Dict with counts of processed, failed, skipped files.
        """
        results = {"processed": 0, "failed": 0, "skipped": 0}

        unprocessed = self._get_unprocessed_files()
        if not unprocessed:
            print("No unprocessed files found.")
            return results

        print(f"Found {len(unprocessed)} unprocessed files")

        for audio_path in unprocessed:
            if not self._is_file_complete(audio_path):
                print(f"Skipping {audio_path.name} (still being written)")
                results["skipped"] += 1
                continue

            if self.process_file(audio_path):
                results["processed"] += 1
            else:
                results["failed"] += 1

        return results

    def start(self) -> None:
        """Start continuous monitoring."""
        self._running = True

        # Write PID file
        self._pid_file.write_text(str(os.getpid()))

        # Setup signal handlers
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        print(f"Job runner started (PID: {os.getpid()})")
        print(f"Uploads folder: {self.uploads_dir}")
        print(f"Processed folder: {self.processed_dir}")
        print(f"Mode: {self.mode}")
        print(f"Language: {self.language or 'auto-detect'}")
        print(f"Check interval: {self.check_interval}s")
        print("Press Ctrl+C to stop")
        print()

        try:
            while self._running:
                unprocessed = self._get_unprocessed_files()

                for audio_path in unprocessed:
                    if not self._running:
                        break

                    if not self._is_file_complete(audio_path):
                        continue

                    self.process_file(audio_path)

                if self._running:
                    time.sleep(self.check_interval)
        finally:
            self._cleanup()

    def stop(self) -> None:
        """Stop monitoring."""
        self._running = False

    def _handle_signal(self, signum, frame) -> None:
        """Handle shutdown signals."""
        print(f"\nReceived signal {signum}, stopping...")
        if self._current_file:
            print(f"(finishing {self._current_file})")
        self.stop()

    def _cleanup(self) -> None:
        """Cleanup on shutdown."""
        if self._pid_file.exists():
            self._pid_file.unlink()
        print("Job runner stopped")

    @classmethod
    def is_running(cls, data_dir: Optional[Path] = None) -> Optional[int]:
        """Check if a job runner is running."""
        settings = get_settings()
        pid_file = Path(data_dir or settings.data_dir) / ".job_runner.pid"
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
    def stop_running(cls, data_dir: Optional[Path] = None) -> bool:
        """Stop a running job runner."""
        pid = cls.is_running(data_dir)
        if pid is None:
            return False

        try:
            os.kill(pid, signal.SIGTERM)
            for _ in range(30):  # Wait up to 15 seconds
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
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Process uploaded audio files")
    parser.add_argument("command", choices=["start", "stop", "status", "process"])
    parser.add_argument("--mode", "-m", choices=["hybrid", "openai", "local"], help="Transcription mode")
    parser.add_argument("--language", "-l", help="Language code (default: auto-detect)")
    parser.add_argument("--interval", "-i", type=int, default=DEFAULT_CHECK_INTERVAL, help="Check interval in seconds")

    args = parser.parse_args()

    if args.command == "status":
        pid = JobRunner.is_running()
        if pid:
            print(f"Job runner is running (PID: {pid})")
        else:
            print("Job runner is not running")
    elif args.command == "stop":
        if JobRunner.stop_running():
            print("Job runner stopped")
        else:
            print("Job runner is not running")
    elif args.command == "start":
        runner = JobRunner(
            mode=args.mode,
            language=args.language,
            check_interval=args.interval,
        )
        runner.start()
    elif args.command == "process":
        runner = JobRunner(
            mode=args.mode,
            language=args.language,
        )
        results = runner.process_backlog()
        print(f"\nResults: {results['processed']} processed, {results['failed']} failed, {results['skipped']} skipped")


if __name__ == "__main__":
    main()
