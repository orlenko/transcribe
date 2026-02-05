"""Data classes for transcription models."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Speaker:
    """A known speaker profile."""

    id: int
    name: str
    created_at: datetime
    updated_at: datetime
    sample_count: int = 0
    notes: Optional[str] = None


@dataclass
class SpeakerEmbedding:
    """Voice embedding for a speaker."""

    id: int
    speaker_id: int
    embedding: bytes
    source_audio_id: Optional[int] = None
    segment_start: Optional[float] = None
    segment_end: Optional[float] = None
    quality_score: Optional[float] = None
    created_at: Optional[datetime] = None


@dataclass
class AudioFile:
    """Metadata for a processed audio file."""

    id: int
    file_path: str
    file_hash: str
    duration_seconds: Optional[float] = None
    recorded_at: Optional[datetime] = None  # When the audio was recorded
    created_at: Optional[datetime] = None   # When the record was created


@dataclass
class Transcript:
    """A complete transcript record."""

    id: int
    audio_id: int
    created_at: datetime
    model_name: str = "large-v3"
    language: Optional[str] = None
    title: Optional[str] = None


@dataclass
class TranscriptSegment:
    """A segment of transcribed audio."""

    id: int
    transcript_id: int
    segment_index: int
    start_time: float
    end_time: float
    text: str
    speaker_label: Optional[str] = None
    speaker_id: Optional[int] = None
    confidence: Optional[float] = None


@dataclass
class DiarizedSegment:
    """A segment with diarization info (before DB storage)."""

    start: float
    end: float
    text: str
    speaker_label: str
    words: list = field(default_factory=list)


@dataclass
class MatchedSpeaker:
    """Result of speaker matching."""

    speaker_label: str
    matched_speaker: Optional[Speaker]
    confidence: float
    embedding: Optional[bytes] = None
