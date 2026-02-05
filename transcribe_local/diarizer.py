"""Pyannote-based speaker diarization and embedding extraction."""

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pyannote.audio import Inference, Model, Pipeline

from .env import load_transcribe_env
from .models import DiarizedSegment


def _get_auth_token_kwargs(token: str) -> dict:
    """Get the correct auth token kwarg for the installed huggingface_hub version.

    Older versions use 'use_auth_token', newer versions use 'token'.
    """
    try:
        import inspect
        sig = inspect.signature(Pipeline.from_pretrained)
        if 'token' in sig.parameters:
            return {'token': token}
        else:
            return {'use_auth_token': token}
    except Exception:
        # Default to older style if we can't determine
        return {'use_auth_token': token}


def detect_device() -> str:
    """Detect the best available compute device.

    Checks CUDA compute capability - PyTorch 2.x requires capability >= 7.0.
    Older GPUs (Pascal, Maxwell) must use CPU.
    """
    if torch.cuda.is_available():
        # Check compute capability - PyTorch 2.x requires >= 7.0
        capability = torch.cuda.get_device_capability()
        if capability[0] >= 7:
            return "cuda"
        else:
            import sys
            gpu_name = torch.cuda.get_device_name(0)
            print(
                f"Warning: {gpu_name} (capability {capability[0]}.{capability[1]}) "
                f"is not supported by PyTorch 2.x (requires >= 7.0). Using CPU.",
                file=sys.stderr
            )
            return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Diarizer:
    """Speaker diarization and embedding extraction using pyannote."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        device: Optional[str] = None,
    ):
        load_transcribe_env()
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        if not self.hf_token:
            raise ValueError(
                "Hugging Face token required. Set HF_TOKEN environment variable "
                "or pass hf_token parameter."
            )
        self.device = device or detect_device()
        self._diarize_pipeline = None
        self._embedding_model = None

    @property
    def diarize_pipeline(self):
        """Lazy-load the diarization pipeline."""
        if self._diarize_pipeline is None:
            # Use pyannote's speaker diarization pipeline directly
            auth_kwargs = _get_auth_token_kwargs(self.hf_token)
            self._diarize_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                **auth_kwargs,
            )
            # Move to device
            self._diarize_pipeline = self._diarize_pipeline.to(torch.device(self.device))
        return self._diarize_pipeline

    @property
    def embedding_model(self):
        """Lazy-load the speaker embedding inference model."""
        if self._embedding_model is None:
            # Use pyannote's speaker embedding model via Inference
            auth_kwargs = _get_auth_token_kwargs(self.hf_token)
            model = Model.from_pretrained(
                "pyannote/embedding",
                **auth_kwargs,
            )
            # Create inference wrapper - handles batching and device placement
            self._embedding_model = Inference(
                model,
                window="whole",
                device=torch.device(self.device),
            )
        return self._embedding_model

    def diarize(
        self,
        audio_path: str,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        """Perform speaker diarization on audio file.

        Args:
            audio_path: Path to audio file.
            min_speakers: Minimum expected number of speakers.
            max_speakers: Maximum expected number of speakers.

        Returns:
            Diarization result as pandas DataFrame with start, end, speaker columns.
        """
        # Run diarization pipeline
        diarization = self.diarize_pipeline(
            audio_path,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )

        # Convert pyannote Annotation to DataFrame for easier processing
        records = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            records.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
            })

        return pd.DataFrame(records)

    def assign_speakers(
        self,
        diarize_segments,
        aligned_result: dict,
    ) -> list[DiarizedSegment]:
        """Assign speaker labels to transcription segments.

        Note: This method is for compatibility with the full local mode.
        For hybrid mode, use _merge_transcription_with_diarization in cli.py.

        Args:
            diarize_segments: Result from diarize() (DataFrame).
            aligned_result: Aligned transcription result from Transcriber.

        Returns:
            List of DiarizedSegment with speaker labels.
        """
        segments = []
        for seg in aligned_result.get("segments", []):
            seg_start = seg["start"]
            seg_end = seg["end"]
            seg_text = seg.get("text", "").strip()

            if not seg_text:
                continue

            # Find dominant speaker for this segment
            speaker = self._find_speaker_for_segment(seg_start, seg_end, diarize_segments)

            segments.append(
                DiarizedSegment(
                    start=seg_start,
                    end=seg_end,
                    text=seg_text,
                    speaker_label=speaker,
                    words=seg.get("words", []),
                )
            )

        return segments

    def _find_speaker_for_segment(self, start: float, end: float, diarize_df) -> str:
        """Find the dominant speaker for a time segment."""
        if diarize_df is None or len(diarize_df) == 0:
            return "UNKNOWN"

        speaker_overlap = {}
        for _, row in diarize_df.iterrows():
            overlap_start = max(start, row["start"])
            overlap_end = min(end, row["end"])
            overlap = max(0, overlap_end - overlap_start)

            if overlap > 0:
                speaker = row["speaker"]
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0) + overlap

        if not speaker_overlap:
            return "UNKNOWN"

        return max(speaker_overlap, key=speaker_overlap.get)

    def extract_embedding(
        self,
        audio_path: str,
        start: float,
        end: float,
    ) -> np.ndarray:
        """Extract speaker embedding from an audio segment.

        Args:
            audio_path: Path to audio file.
            start: Start time in seconds.
            end: End time in seconds.

        Returns:
            Speaker embedding as numpy array.
        """
        from pyannote.core import Segment

        # Use Inference to extract embedding from segment
        # Inference handles audio loading and device placement
        segment = Segment(start, end)
        embedding = self.embedding_model.crop(audio_path, segment)

        return embedding.flatten().astype(np.float32)

    def extract_embeddings_for_speaker(
        self,
        audio_path: str,
        segments: list[DiarizedSegment],
        speaker_label: str,
        min_duration: float = 1.0,
        max_embeddings: int = 10,
    ) -> list[tuple[np.ndarray, float, float, float]]:
        """Extract embeddings from all segments for a given speaker.

        Args:
            audio_path: Path to audio file.
            segments: List of DiarizedSegment objects.
            speaker_label: Speaker label to extract embeddings for.
            min_duration: Minimum segment duration to use.
            max_embeddings: Maximum number of embeddings to extract.

        Returns:
            List of (embedding, start, end, quality_score) tuples.
        """
        # Filter segments for this speaker
        speaker_segments = [
            seg for seg in segments
            if seg.speaker_label == speaker_label and (seg.end - seg.start) >= min_duration
        ]

        # Sort by duration (longer = better quality)
        speaker_segments.sort(key=lambda s: s.end - s.start, reverse=True)

        # Take top segments
        speaker_segments = speaker_segments[:max_embeddings]

        embeddings = []
        for seg in speaker_segments:
            try:
                embedding = self.extract_embedding(audio_path, seg.start, seg.end)
                duration = seg.end - seg.start
                # Quality score based on duration (longer is better, capped at 30s)
                quality_score = min(duration / 30.0, 1.0)
                embeddings.append((embedding, seg.start, seg.end, quality_score))
            except Exception as e:
                # Skip segments that fail
                print(f"Warning: Could not extract embedding for segment {seg.start}-{seg.end}: {e}")
                continue

        return embeddings

    def get_unique_speakers(self, segments: list[DiarizedSegment]) -> list[str]:
        """Get unique speaker labels from segments.

        Args:
            segments: List of DiarizedSegment objects.

        Returns:
            List of unique speaker labels.
        """
        seen = set()
        speakers = []
        for seg in segments:
            if seg.speaker_label not in seen:
                seen.add(seg.speaker_label)
                speakers.append(seg.speaker_label)
        return speakers
