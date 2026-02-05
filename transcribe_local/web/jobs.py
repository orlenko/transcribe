"""Job management for background transcription tasks."""

import os
import tempfile
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from ..database import Database
from ..env import load_transcribe_env
from ..models import DiarizedSegment
from ..timezone_utils import format_local_iso, utc_now
from .config import Settings


class JobStatus(str, Enum):
    """Job status enum."""
    QUEUED = "queued"
    TRANSCRIBING = "transcribing"
    DIARIZING = "diarizing"
    MERGING = "merging"
    MATCHING = "matching"
    SAVING = "saving"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Transcription job."""
    id: str
    file_path: str
    original_filename: str
    mode: str  # "hybrid", "openai", "local"
    language: Optional[str] = None
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None

    status: JobStatus = JobStatus.QUEUED
    progress: float = 0.0
    message: str = ""
    error: Optional[str] = None

    transcript_id: Optional[int] = None
    created_at: datetime = field(default_factory=utc_now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "file_path": self.file_path,
            "original_filename": self.original_filename,
            "mode": self.mode,
            "language": self.language,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "transcript_id": self.transcript_id,
            "created_at": format_local_iso(self.created_at),
            "started_at": format_local_iso(self.started_at),
            "completed_at": format_local_iso(self.completed_at),
        }


class JobManager:
    """Manages transcription jobs."""

    def __init__(self, db: Database, settings: Settings):
        self.db = db
        self.settings = settings
        self._jobs: dict[str, Job] = {}

    def create_job(
        self,
        file_path: str,
        original_filename: str,
        mode: str,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> str:
        """Create a new transcription job."""
        job_id = str(uuid.uuid4())[:8]

        job = Job(
            id=job_id,
            file_path=file_path,
            original_filename=original_filename,
            mode=mode,
            language=language or self.settings.default_language,
            min_speakers=min_speakers or self.settings.min_speakers,
            max_speakers=max_speakers or self.settings.max_speakers,
        )

        self._jobs[job_id] = job
        return job_id

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get a job by ID."""
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        """List all jobs, most recent first."""
        return sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)

    def update_job(
        self,
        job_id: str,
        status: Optional[JobStatus] = None,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error: Optional[str] = None,
        transcript_id: Optional[int] = None,
    ) -> None:
        """Update job status."""
        job = self._jobs.get(job_id)
        if not job:
            return

        if status is not None:
            job.status = status
            if status == JobStatus.TRANSCRIBING and not job.started_at:
                job.started_at = utc_now()
            elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                job.completed_at = utc_now()

        if progress is not None:
            job.progress = progress
        if message is not None:
            job.message = message
        if error is not None:
            job.error = error
        if transcript_id is not None:
            job.transcript_id = transcript_id

    async def run_job(self, job_id: str) -> None:
        """Run a transcription job."""
        job = self._jobs.get(job_id)
        if not job:
            return

        try:
            if job.mode == "openai":
                await self._run_openai_only(job)
            elif job.mode == "local":
                await self._run_local(job)
            else:  # hybrid
                await self._run_hybrid(job)

        except Exception as e:
            self.update_job(
                job_id,
                status=JobStatus.FAILED,
                error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
            )

    async def _run_openai_only(self, job: Job) -> None:
        """Run OpenAI-only transcription (no diarization)."""
        import openai

        load_transcribe_env()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        openai.api_key = api_key

        self.update_job(job.id, status=JobStatus.TRANSCRIBING, progress=0.1, message="Transcribing with OpenAI...")

        # Check if chunking needed
        file_size = os.path.getsize(job.file_path)
        max_size = 24 * 1024 * 1024

        if file_size > max_size:
            segments, detected_lang = await self._transcribe_openai_chunked(job)
        else:
            with open(job.file_path, "rb") as f:
                result = openai.audio.transcriptions.create(
                    model=self.settings.openai_model,
                    file=f,
                    language=job.language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment"],
                )

            detected_lang = result.language if hasattr(result, "language") else job.language or "en"
            segments = []
            if hasattr(result, "segments") and result.segments:
                for seg in result.segments:
                    segments.append({
                        "start": seg.start if hasattr(seg, "start") else seg["start"],
                        "end": seg.end if hasattr(seg, "end") else seg["end"],
                        "text": seg.text if hasattr(seg, "text") else seg["text"],
                    })

        self.update_job(job.id, status=JobStatus.SAVING, progress=0.9, message="Saving to database...")

        # Save to database (no speaker labels in openai-only mode)
        audio_record = self.db.add_audio_file(job.file_path)
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
                speaker_label="SPEAKER_00",  # Single speaker assumed
            )

        self.update_job(
            job.id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            message="Completed",
            transcript_id=transcript_record.id,
        )

    async def _run_hybrid(self, job: Job) -> None:
        """Run hybrid transcription (OpenAI + local diarization)."""
        import openai

        from ..diarizer import Diarizer
        from ..speaker_matcher import SpeakerMatcher, format_match_result, update_segments_with_matches

        load_transcribe_env()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        openai.api_key = api_key

        # Step 1: Transcribe with OpenAI
        self.update_job(job.id, status=JobStatus.TRANSCRIBING, progress=0.1, message="Transcribing with OpenAI...")

        file_size = os.path.getsize(job.file_path)
        max_size = 24 * 1024 * 1024

        if file_size > max_size:
            openai_segments, detected_lang = await self._transcribe_openai_chunked(job)
        else:
            with open(job.file_path, "rb") as f:
                result = openai.audio.transcriptions.create(
                    model=self.settings.openai_model,
                    file=f,
                    language=job.language,
                    response_format="verbose_json",
                    timestamp_granularities=["segment", "word"],
                )

            detected_lang = result.language if hasattr(result, "language") else job.language or "en"
            openai_segments = []
            if hasattr(result, "segments") and result.segments:
                for seg in result.segments:
                    openai_segments.append({
                        "start": seg.start if hasattr(seg, "start") else seg["start"],
                        "end": seg.end if hasattr(seg, "end") else seg["end"],
                        "text": seg.text if hasattr(seg, "text") else seg["text"],
                    })

        self.update_job(job.id, progress=0.4, message=f"Got {len(openai_segments)} segments")

        # Step 2: Diarize locally
        self.update_job(job.id, status=JobStatus.DIARIZING, progress=0.5, message="Performing speaker diarization...")

        diarizer = Diarizer()
        diarize_result = diarizer.diarize(
            job.file_path,
            min_speakers=job.min_speakers,
            max_speakers=job.max_speakers,
        )

        # Step 3: Merge
        self.update_job(job.id, status=JobStatus.MERGING, progress=0.7, message="Merging transcription with speakers...")

        segments = self._merge_transcription_with_diarization(openai_segments, diarize_result)
        unique_speakers = diarizer.get_unique_speakers(segments)

        # Step 4: Match speakers
        self.update_job(job.id, status=JobStatus.MATCHING, progress=0.8, message="Matching known speakers...")

        matcher = SpeakerMatcher(self.db, threshold=self.settings.speaker_match_threshold)
        matches = []

        speaker_embeddings = {}
        for speaker_label in unique_speakers:
            embeddings = diarizer.extract_embeddings_for_speaker(
                job.file_path, segments, speaker_label, max_embeddings=1
            )
            if embeddings:
                speaker_embeddings[speaker_label] = embeddings[0][0]

        if speaker_embeddings:
            matches = matcher.match_speakers(speaker_embeddings)
            segments = update_segments_with_matches(segments, matches)

        # Step 5: Save
        self.update_job(job.id, status=JobStatus.SAVING, progress=0.9, message="Saving to database...")

        audio_record = self.db.add_audio_file(job.file_path)
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

        self.update_job(
            job.id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            message=f"Completed - {len(unique_speakers)} speakers detected",
            transcript_id=transcript_record.id,
        )

    async def _run_local(self, job: Job) -> None:
        """Run fully local transcription (WhisperX + pyannote)."""
        from ..diarizer import Diarizer
        from ..speaker_matcher import SpeakerMatcher, update_segments_with_matches
        from ..transcriber import Transcriber

        # Step 1: Transcribe locally
        self.update_job(job.id, status=JobStatus.TRANSCRIBING, progress=0.1, message="Loading local models...")

        transcriber = Transcriber(model_name=self.settings.local_model)
        diarizer = Diarizer()

        self.update_job(job.id, progress=0.2, message="Transcribing audio (local)...")

        aligned_result, detected_lang = transcriber.transcribe_and_align(
            job.file_path,
            language=job.language,
        )

        self.update_job(job.id, status=JobStatus.DIARIZING, progress=0.5, message="Performing speaker diarization...")

        diarize_result = diarizer.diarize(
            job.file_path,
            min_speakers=job.min_speakers,
            max_speakers=job.max_speakers,
        )

        self.update_job(job.id, status=JobStatus.MERGING, progress=0.7, message="Assigning speakers...")

        segments = diarizer.assign_speakers(diarize_result, aligned_result)
        unique_speakers = diarizer.get_unique_speakers(segments)

        # Match speakers
        self.update_job(job.id, status=JobStatus.MATCHING, progress=0.8, message="Matching known speakers...")

        matcher = SpeakerMatcher(self.db, threshold=self.settings.speaker_match_threshold)
        matches = []

        speaker_embeddings = {}
        for speaker_label in unique_speakers:
            embeddings = diarizer.extract_embeddings_for_speaker(
                job.file_path, segments, speaker_label, max_embeddings=1
            )
            if embeddings:
                speaker_embeddings[speaker_label] = embeddings[0][0]

        if speaker_embeddings:
            matches = matcher.match_speakers(speaker_embeddings)
            segments = update_segments_with_matches(segments, matches)

        # Save
        self.update_job(job.id, status=JobStatus.SAVING, progress=0.9, message="Saving to database...")

        audio_record = self.db.add_audio_file(job.file_path)
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

        self.update_job(
            job.id,
            status=JobStatus.COMPLETED,
            progress=1.0,
            message=f"Completed - {len(unique_speakers)} speakers detected",
            transcript_id=transcript_record.id,
        )

    async def _transcribe_openai_chunked(self, job: Job) -> tuple[list[dict], str]:
        """Transcribe large audio file in chunks."""
        import openai
        from pydub import AudioSegment

        self.update_job(job.id, message="Loading audio for chunking...")

        audio = AudioSegment.from_file(job.file_path)
        total_duration_ms = len(audio)
        chunk_duration_ms = 10 * 60 * 1000  # 10 minutes

        chunks = []
        start_ms = 0
        while start_ms < total_duration_ms:
            end_ms = min(start_ms + chunk_duration_ms, total_duration_ms)
            chunks.append((start_ms, end_ms))
            start_ms = end_ms

        all_segments = []
        detected_lang = job.language or "en"

        for i, (start_ms, end_ms) in enumerate(chunks):
            chunk_start_sec = start_ms / 1000
            progress = 0.1 + (0.3 * (i + 1) / len(chunks))

            self.update_job(
                job.id,
                progress=progress,
                message=f"Transcribing chunk {i + 1}/{len(chunks)}...",
            )

            chunk_audio = audio[start_ms:end_ms]

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_path = tmp.name
                chunk_audio.export(tmp_path, format="mp3")

            try:
                with open(tmp_path, "rb") as f:
                    result = openai.audio.transcriptions.create(
                        model=self.settings.openai_model,
                        file=f,
                        language=job.language,
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
