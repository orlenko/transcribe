"""SQLite database operations for speaker-aware transcription.

Timestamp handling:
- SQLite CURRENT_TIMESTAMP stores UTC time
- All timestamps are parsed as UTC when reading from database
- Display conversion to local time happens in presentation layer
"""

import hashlib
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from .models import (
    AudioFile,
    Speaker,
    SpeakerEmbedding,
    Transcript,
    TranscriptSegment,
)
from .paths import get_data_dir
from .timezone_utils import parse_utc_timestamp

DEFAULT_DB_PATH = get_data_dir() / "speakers.db"

SCHEMA = """
-- Speaker profiles
CREATE TABLE IF NOT EXISTS speakers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    sample_count INTEGER DEFAULT 0,
    notes TEXT
);

-- Voice embeddings (multiple per speaker for averaging)
CREATE TABLE IF NOT EXISTS speaker_embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    source_audio_id INTEGER,
    segment_start REAL,
    segment_end REAL,
    quality_score REAL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE CASCADE
);

-- Audio file metadata
CREATE TABLE IF NOT EXISTS audio_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path TEXT NOT NULL,
    file_hash TEXT NOT NULL UNIQUE,
    duration_seconds REAL,
    recorded_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transcripts
CREATE TABLE IF NOT EXISTS transcripts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    audio_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_name TEXT DEFAULT 'large-v3',
    language TEXT,
    title TEXT,
    FOREIGN KEY (audio_id) REFERENCES audio_files(id) ON DELETE CASCADE
);

-- Transcript segments with speaker labels
CREATE TABLE IF NOT EXISTS transcript_segments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    transcript_id INTEGER NOT NULL,
    segment_index INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    text TEXT NOT NULL,
    speaker_label TEXT,
    speaker_id INTEGER,
    confidence REAL,
    FOREIGN KEY (transcript_id) REFERENCES transcripts(id) ON DELETE CASCADE,
    FOREIGN KEY (speaker_id) REFERENCES speakers(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_transcript ON transcript_segments(transcript_id);
CREATE INDEX IF NOT EXISTS idx_segments_speaker ON transcript_segments(speaker_id);
CREATE INDEX IF NOT EXISTS idx_embeddings_speaker ON speaker_embeddings(speaker_id);
"""


class Database:
    """SQLite database for speaker profiles and transcripts."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            # Migration: add title column if it doesn't exist
            try:
                conn.execute("ALTER TABLE transcripts ADD COLUMN title TEXT")
            except sqlite3.OperationalError:
                pass  # Column already exists
            # Migration: add recorded_at column if it doesn't exist
            try:
                conn.execute("ALTER TABLE audio_files ADD COLUMN recorded_at TIMESTAMP")
            except sqlite3.OperationalError:
                pass  # Column already exists

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # Speaker operations

    def add_speaker(self, name: str, notes: Optional[str] = None) -> Speaker:
        """Add a new speaker to the database."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO speakers (name, notes) VALUES (?, ?)",
                (name, notes),
            )
            speaker_id = cursor.lastrowid
            row = conn.execute(
                "SELECT * FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
            return self._row_to_speaker(row)

    def get_speaker(self, name: str) -> Optional[Speaker]:
        """Get a speaker by name."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM speakers WHERE name = ?", (name,)
            ).fetchone()
            return self._row_to_speaker(row) if row else None

    def get_speaker_by_id(self, speaker_id: int) -> Optional[Speaker]:
        """Get a speaker by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM speakers WHERE id = ?", (speaker_id,)
            ).fetchone()
            return self._row_to_speaker(row) if row else None

    def list_speakers(self) -> list[Speaker]:
        """List all speakers."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM speakers ORDER BY name"
            ).fetchall()
            return [self._row_to_speaker(row) for row in rows]

    def delete_speaker(self, name: str) -> bool:
        """Delete a speaker by name."""
        with self._connect() as conn:
            cursor = conn.execute("DELETE FROM speakers WHERE name = ?", (name,))
            return cursor.rowcount > 0

    def rename_speaker(self, old_name: str, new_name: str) -> bool:
        """Rename a speaker."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE speakers SET name = ?, updated_at = CURRENT_TIMESTAMP WHERE name = ?",
                (new_name, old_name),
            )
            return cursor.rowcount > 0

    def update_speaker_sample_count(self, speaker_id: int) -> None:
        """Update the sample count for a speaker."""
        with self._connect() as conn:
            conn.execute(
                """UPDATE speakers
                   SET sample_count = (SELECT COUNT(*) FROM speaker_embeddings WHERE speaker_id = ?),
                       updated_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (speaker_id, speaker_id),
            )

    def _row_to_speaker(self, row: sqlite3.Row) -> Speaker:
        """Convert a database row to a Speaker object."""
        return Speaker(
            id=row["id"],
            name=row["name"],
            created_at=parse_utc_timestamp(row["created_at"]),
            updated_at=parse_utc_timestamp(row["updated_at"]),
            sample_count=row["sample_count"],
            notes=row["notes"],
        )

    # Embedding operations

    def add_embedding(
        self,
        speaker_id: int,
        embedding: np.ndarray,
        source_audio_id: Optional[int] = None,
        segment_start: Optional[float] = None,
        segment_end: Optional[float] = None,
        quality_score: Optional[float] = None,
    ) -> int:
        """Add a voice embedding for a speaker."""
        embedding_bytes = embedding.tobytes()
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO speaker_embeddings
                   (speaker_id, embedding, source_audio_id, segment_start, segment_end, quality_score)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (speaker_id, embedding_bytes, source_audio_id, segment_start, segment_end, quality_score),
            )
            self.update_speaker_sample_count(speaker_id)
            return cursor.lastrowid

    def get_speaker_embeddings(self, speaker_id: int) -> list[np.ndarray]:
        """Get all embeddings for a speaker."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT embedding FROM speaker_embeddings WHERE speaker_id = ?",
                (speaker_id,),
            ).fetchall()
            return [np.frombuffer(row["embedding"], dtype=np.float32) for row in rows]

    def get_all_speaker_embeddings(self) -> dict[int, list[np.ndarray]]:
        """Get all embeddings grouped by speaker ID."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT speaker_id, embedding FROM speaker_embeddings"
            ).fetchall()
            result: dict[int, list[np.ndarray]] = {}
            for row in rows:
                speaker_id = row["speaker_id"]
                embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                if speaker_id not in result:
                    result[speaker_id] = []
                result[speaker_id].append(embedding)
            return result

    def count_speaker_embeddings(self, speaker_id: int) -> int:
        """Count embeddings for a speaker."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as count FROM speaker_embeddings WHERE speaker_id = ?",
                (speaker_id,),
            ).fetchone()
            return row["count"]

    def delete_oldest_embeddings(self, speaker_id: int, keep_count: int = 50) -> int:
        """Delete oldest embeddings, keeping only the most recent ones."""
        with self._connect() as conn:
            cursor = conn.execute(
                """DELETE FROM speaker_embeddings
                   WHERE speaker_id = ? AND id NOT IN (
                       SELECT id FROM speaker_embeddings
                       WHERE speaker_id = ?
                       ORDER BY created_at DESC
                       LIMIT ?
                   )""",
                (speaker_id, speaker_id, keep_count),
            )
            if cursor.rowcount > 0:
                self.update_speaker_sample_count(speaker_id)
            return cursor.rowcount

    # Audio file operations

    def add_audio_file(
        self,
        file_path: str,
        duration_seconds: Optional[float] = None,
        recorded_at: Optional[datetime] = None,
    ) -> AudioFile:
        """Add an audio file record."""
        file_hash = self._compute_file_hash(file_path)
        with self._connect() as conn:
            # Check if file already exists
            existing = conn.execute(
                "SELECT * FROM audio_files WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            if existing:
                return self._row_to_audio_file(existing)

            # Convert recorded_at to string for storage
            recorded_at_str = None
            if recorded_at:
                from .timezone_utils import to_utc_string
                recorded_at_str = to_utc_string(recorded_at)

            cursor = conn.execute(
                "INSERT INTO audio_files (file_path, file_hash, duration_seconds, recorded_at) VALUES (?, ?, ?, ?)",
                (file_path, file_hash, duration_seconds, recorded_at_str),
            )
            row = conn.execute(
                "SELECT * FROM audio_files WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            return self._row_to_audio_file(row)

    def get_audio_file_by_hash(self, file_hash: str) -> Optional[AudioFile]:
        """Get an audio file by its hash."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM audio_files WHERE file_hash = ?", (file_hash,)
            ).fetchone()
            return self._row_to_audio_file(row) if row else None

    def _compute_file_hash(self, file_path: str) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def _row_to_audio_file(self, row: sqlite3.Row) -> AudioFile:
        """Convert a database row to an AudioFile object."""
        # Handle recorded_at which may not exist in older databases
        recorded_at = None
        if "recorded_at" in row.keys() and row["recorded_at"]:
            recorded_at = parse_utc_timestamp(row["recorded_at"])
        return AudioFile(
            id=row["id"],
            file_path=row["file_path"],
            file_hash=row["file_hash"],
            duration_seconds=row["duration_seconds"],
            recorded_at=recorded_at,
            created_at=parse_utc_timestamp(row["created_at"]),
        )

    # Transcript operations

    def add_transcript(
        self,
        audio_id: int,
        model_name: str = "large-v3",
        language: Optional[str] = None,
        title: Optional[str] = None,
    ) -> Transcript:
        """Add a transcript record."""
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO transcripts (audio_id, model_name, language, title) VALUES (?, ?, ?, ?)",
                (audio_id, model_name, language, title),
            )
            row = conn.execute(
                "SELECT * FROM transcripts WHERE id = ?", (cursor.lastrowid,)
            ).fetchone()
            return self._row_to_transcript(row)

    def get_transcript(self, transcript_id: int) -> Optional[Transcript]:
        """Get a transcript by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM transcripts WHERE id = ?", (transcript_id,)
            ).fetchone()
            return self._row_to_transcript(row) if row else None

    def list_transcripts(
        self, limit: Optional[int] = None, offset: int = 0
    ) -> list[tuple[Transcript, AudioFile]]:
        """List transcripts with their audio file info.

        Args:
            limit: Maximum number of transcripts to return (None for all).
            offset: Number of transcripts to skip.

        Returns:
            List of (Transcript, AudioFile) tuples.
        """
        with self._connect() as conn:
            if limit is not None:
                rows = conn.execute(
                    """SELECT t.*, a.file_path, a.file_hash, a.duration_seconds,
                              a.recorded_at as audio_recorded_at, a.created_at as audio_created_at
                       FROM transcripts t
                       JOIN audio_files a ON t.audio_id = a.id
                       ORDER BY COALESCE(a.recorded_at, t.created_at) DESC
                       LIMIT ? OFFSET ?""",
                    (limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """SELECT t.*, a.file_path, a.file_hash, a.duration_seconds,
                              a.recorded_at as audio_recorded_at, a.created_at as audio_created_at
                       FROM transcripts t
                       JOIN audio_files a ON t.audio_id = a.id
                       ORDER BY COALESCE(a.recorded_at, t.created_at) DESC"""
                ).fetchall()
            result = []
            for row in rows:
                transcript = self._row_to_transcript(row)
                recorded_at = None
                if "audio_recorded_at" in row.keys() and row["audio_recorded_at"]:
                    recorded_at = parse_utc_timestamp(row["audio_recorded_at"])
                audio = AudioFile(
                    id=row["audio_id"],
                    file_path=row["file_path"],
                    file_hash=row["file_hash"],
                    duration_seconds=row["duration_seconds"],
                    recorded_at=recorded_at,
                    created_at=parse_utc_timestamp(row["audio_created_at"]),
                )
                result.append((transcript, audio))
            return result

    def count_transcripts(self) -> int:
        """Count total number of transcripts."""
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) as count FROM transcripts").fetchone()
            return row["count"]

    def delete_transcript(self, transcript_id: int) -> bool:
        """Delete a transcript."""
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM transcripts WHERE id = ?", (transcript_id,)
            )
            return cursor.rowcount > 0

    def update_transcript_title(self, transcript_id: int, title: str) -> bool:
        """Update the title of a transcript."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE transcripts SET title = ? WHERE id = ?",
                (title, transcript_id),
            )
            return cursor.rowcount > 0

    def _row_to_transcript(self, row: sqlite3.Row) -> Transcript:
        """Convert a database row to a Transcript object."""
        return Transcript(
            id=row["id"],
            audio_id=row["audio_id"],
            created_at=parse_utc_timestamp(row["created_at"]),
            model_name=row["model_name"],
            language=row["language"],
            title=row["title"] if "title" in row.keys() else None,
        )

    # Segment operations

    def add_segment(
        self,
        transcript_id: int,
        segment_index: int,
        start_time: float,
        end_time: float,
        text: str,
        speaker_label: Optional[str] = None,
        speaker_id: Optional[int] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Add a transcript segment."""
        with self._connect() as conn:
            cursor = conn.execute(
                """INSERT INTO transcript_segments
                   (transcript_id, segment_index, start_time, end_time, text, speaker_label, speaker_id, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (transcript_id, segment_index, start_time, end_time, text, speaker_label, speaker_id, confidence),
            )
            return cursor.lastrowid

    def get_segments(self, transcript_id: int) -> list[TranscriptSegment]:
        """Get all segments for a transcript."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM transcript_segments
                   WHERE transcript_id = ?
                   ORDER BY segment_index""",
                (transcript_id,),
            ).fetchall()
            return [self._row_to_segment(row) for row in rows]

    def get_segments_by_speaker_label(
        self, transcript_id: int, speaker_label: str
    ) -> list[TranscriptSegment]:
        """Get segments for a specific speaker label."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT * FROM transcript_segments
                   WHERE transcript_id = ? AND speaker_label = ?
                   ORDER BY segment_index""",
                (transcript_id, speaker_label),
            ).fetchall()
            return [self._row_to_segment(row) for row in rows]

    def update_segment_speaker(
        self, segment_id: int, speaker_id: int, confidence: Optional[float] = None
    ) -> bool:
        """Update the speaker assignment for a segment."""
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE transcript_segments SET speaker_id = ?, confidence = ? WHERE id = ?",
                (speaker_id, confidence, segment_id),
            )
            return cursor.rowcount > 0

    def assign_speaker_to_label(
        self, transcript_id: int, speaker_label: str, speaker_id: int
    ) -> int:
        """Assign a speaker to all segments with a given label."""
        with self._connect() as conn:
            cursor = conn.execute(
                """UPDATE transcript_segments
                   SET speaker_id = ?
                   WHERE transcript_id = ? AND speaker_label = ?""",
                (speaker_id, transcript_id, speaker_label),
            )
            return cursor.rowcount

    def _row_to_segment(self, row: sqlite3.Row) -> TranscriptSegment:
        """Convert a database row to a TranscriptSegment object."""
        return TranscriptSegment(
            id=row["id"],
            transcript_id=row["transcript_id"],
            segment_index=row["segment_index"],
            start_time=row["start_time"],
            end_time=row["end_time"],
            text=row["text"],
            speaker_label=row["speaker_label"],
            speaker_id=row["speaker_id"],
            confidence=row["confidence"],
        )
