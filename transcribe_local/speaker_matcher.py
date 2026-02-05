"""Speaker identification and matching logic."""

from typing import Optional

import numpy as np

from .database import Database
from .models import DiarizedSegment, MatchedSpeaker, Speaker


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity (1.0 = identical, 0.0 = orthogonal, -1.0 = opposite).
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine distance (0.0 = identical, 1.0 = orthogonal, 2.0 = opposite).
    """
    return 1.0 - cosine_similarity(a, b)


class SpeakerMatcher:
    """Match unknown speakers to known speaker profiles."""

    def __init__(
        self,
        db: Database,
        threshold: float = 0.5,
        max_embeddings_per_speaker: int = 50,
    ):
        """Initialize speaker matcher.

        Args:
            db: Database instance.
            threshold: Cosine distance threshold for matching (lower = stricter).
            max_embeddings_per_speaker: Maximum embeddings to store per speaker.
        """
        self.db = db
        self.threshold = threshold
        self.max_embeddings_per_speaker = max_embeddings_per_speaker
        self._speaker_centroids: Optional[dict[int, np.ndarray]] = None

    def compute_centroid(self, embeddings: list[np.ndarray]) -> np.ndarray:
        """Compute centroid (average) of embeddings.

        Args:
            embeddings: List of embedding vectors.

        Returns:
            Centroid vector.
        """
        if not embeddings:
            raise ValueError("Cannot compute centroid of empty list")
        stacked = np.stack(embeddings)
        return np.mean(stacked, axis=0)

    def load_speaker_centroids(self) -> dict[int, np.ndarray]:
        """Load and compute centroids for all known speakers.

        Returns:
            Dict mapping speaker ID to centroid embedding.
        """
        all_embeddings = self.db.get_all_speaker_embeddings()
        centroids = {}
        for speaker_id, embeddings in all_embeddings.items():
            if embeddings:
                centroids[speaker_id] = self.compute_centroid(embeddings)
        self._speaker_centroids = centroids
        return centroids

    def match_embedding(
        self,
        embedding: np.ndarray,
        centroids: Optional[dict[int, np.ndarray]] = None,
    ) -> tuple[Optional[int], float]:
        """Match an embedding to the closest known speaker.

        Args:
            embedding: Embedding to match.
            centroids: Pre-computed centroids (loads from DB if None).

        Returns:
            Tuple of (speaker_id, distance). speaker_id is None if no match
            within threshold.
        """
        if centroids is None:
            centroids = self._speaker_centroids or self.load_speaker_centroids()

        if not centroids:
            return None, float("inf")

        best_speaker_id = None
        best_distance = float("inf")

        for speaker_id, centroid in centroids.items():
            distance = cosine_distance(embedding, centroid)
            if distance < best_distance:
                best_distance = distance
                best_speaker_id = speaker_id

        if best_distance <= self.threshold:
            return best_speaker_id, best_distance
        return None, best_distance

    def match_speakers(
        self,
        speaker_embeddings: dict[str, np.ndarray],
    ) -> list[MatchedSpeaker]:
        """Match multiple speaker labels to known speakers.

        Args:
            speaker_embeddings: Dict mapping speaker labels to embeddings.

        Returns:
            List of MatchedSpeaker results.
        """
        centroids = self.load_speaker_centroids()
        results = []

        for label, embedding in speaker_embeddings.items():
            speaker_id, distance = self.match_embedding(embedding, centroids)

            matched_speaker = None
            if speaker_id is not None:
                matched_speaker = self.db.get_speaker_by_id(speaker_id)

            confidence = 1.0 - distance if distance < float("inf") else 0.0

            results.append(
                MatchedSpeaker(
                    speaker_label=label,
                    matched_speaker=matched_speaker,
                    confidence=confidence,
                    embedding=embedding.tobytes(),
                )
            )

        return results

    def learn_speaker(
        self,
        speaker_id: int,
        embeddings: list[tuple[np.ndarray, float, float, float]],
        audio_id: Optional[int] = None,
    ) -> int:
        """Store new embeddings for a speaker.

        Args:
            speaker_id: ID of the speaker to learn.
            embeddings: List of (embedding, start, end, quality_score) tuples.
            audio_id: Optional audio file ID for tracking.

        Returns:
            Number of embeddings added.
        """
        count = 0
        for embedding, start, end, quality_score in embeddings:
            self.db.add_embedding(
                speaker_id=speaker_id,
                embedding=embedding,
                source_audio_id=audio_id,
                segment_start=start,
                segment_end=end,
                quality_score=quality_score,
            )
            count += 1

        # Prune old embeddings if we exceed the limit
        current_count = self.db.count_speaker_embeddings(speaker_id)
        if current_count > self.max_embeddings_per_speaker:
            self.db.delete_oldest_embeddings(speaker_id, self.max_embeddings_per_speaker)

        # Invalidate cached centroids
        self._speaker_centroids = None

        return count


def format_match_result(match: MatchedSpeaker) -> str:
    """Format a match result for display.

    Args:
        match: MatchedSpeaker result.

    Returns:
        Formatted string.
    """
    if match.matched_speaker:
        return (
            f"{match.speaker_label} -> {match.matched_speaker.name} "
            f"(confidence: {match.confidence:.1%})"
        )
    return f"{match.speaker_label} -> Unknown"


def update_segments_with_matches(
    segments: list[DiarizedSegment],
    matches: list[MatchedSpeaker],
) -> list[DiarizedSegment]:
    """Update segment speaker labels with matched names.

    Args:
        segments: Original segments with SPEAKER_XX labels.
        matches: Match results from SpeakerMatcher.

    Returns:
        Updated segments with speaker names where matched.
    """
    # Build mapping from label to matched name
    label_to_name = {}
    for match in matches:
        if match.matched_speaker:
            label_to_name[match.speaker_label] = match.matched_speaker.name

    # Update segments
    updated = []
    for seg in segments:
        new_label = label_to_name.get(seg.speaker_label, seg.speaker_label)
        updated.append(
            DiarizedSegment(
                start=seg.start,
                end=seg.end,
                text=seg.text,
                speaker_label=new_label,
                words=seg.words,
            )
        )

    return updated
