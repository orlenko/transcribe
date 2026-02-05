"""WhisperX-based transcription module."""

import os
from typing import Optional

import torch
import whisperx

from .env import load_transcribe_env
from .models import DiarizedSegment


def detect_device() -> str:
    """Detect the best available compute device for WhisperX.

    Note: ctranslate2 (used by whisperx) only supports CPU and CUDA,
    not MPS. So on Apple Silicon, we must use CPU for transcription.

    Also checks CUDA compute capability - PyTorch 2.x requires capability >= 7.0.
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
    # MPS is not supported by ctranslate2, fall back to CPU
    return "cpu"


def get_compute_type(device: str) -> str:
    """Get appropriate compute type for the device.

    Uses int8 for CUDA as it's more universally supported across GPU generations.
    float16 requires compute capability 7.0+ (Volta/Turing/Ampere).
    """
    if device == "cuda":
        # Check if GPU supports float16 efficiently (compute capability >= 7.0)
        if torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            if capability[0] >= 7:
                return "float16"
        # Fall back to int8 for older GPUs
        return "int8"
    # CPU works with int8
    return "int8"


class Transcriber:
    """WhisperX-based audio transcriber."""

    def __init__(
        self,
        model_name: str = "large-v3",
        device: Optional[str] = None,
        hf_token: Optional[str] = None,
    ):
        load_transcribe_env()
        self.model_name = model_name
        self.device = device or detect_device()
        self.compute_type = get_compute_type(self.device)
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")

        self._model = None
        self._diarize_model = None
        self._align_model = None
        self._align_metadata = None

    @property
    def model(self):
        """Lazy-load the WhisperX model."""
        if self._model is None:
            self._model = whisperx.load_model(
                self.model_name,
                self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        batch_size: int = 16,
    ) -> dict:
        """Transcribe audio file.

        Args:
            audio_path: Path to audio file.
            language: Language code (auto-detect if None).
            batch_size: Batch size for transcription.

        Returns:
            Transcription result dict with segments.
        """
        # Load audio
        audio = whisperx.load_audio(audio_path)

        # Transcribe
        result = self.model.transcribe(
            audio,
            batch_size=batch_size,
            language=language,
        )

        return result, audio

    def align(self, result: dict, audio, language: str) -> dict:
        """Align transcription with audio for word-level timestamps.

        Args:
            result: Transcription result from transcribe().
            audio: Audio array from transcribe().
            language: Detected or specified language.

        Returns:
            Aligned transcription result.
        """
        # Load alignment model if needed
        if self._align_model is None or self._current_align_lang != language:
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language,
                device=self.device,
            )
            self._current_align_lang = language

        # Align
        result = whisperx.align(
            result["segments"],
            self._align_model,
            self._align_metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        return result

    def transcribe_and_align(
        self,
        audio_path: str,
        language: Optional[str] = None,
        batch_size: int = 16,
        fallback_language: str = "en",
        min_language_confidence: float = 0.7,
    ) -> tuple[dict, str]:
        """Transcribe and align audio in one call.

        Args:
            audio_path: Path to audio file.
            language: Language code (auto-detect if None).
            batch_size: Batch size for transcription.
            fallback_language: Language to use if detection fails or confidence is low.
            min_language_confidence: Minimum confidence for detected language.

        Returns:
            Tuple of (aligned result, detected language).
        """
        result, audio = self.transcribe(audio_path, language, batch_size)

        # Get detected language and confidence
        detected_lang = result.get("language", language or fallback_language)
        lang_confidence = result.get("language_probability", 1.0)

        # Common/supported languages for alignment
        supported_languages = {
            "en", "es", "fr", "de", "it", "pt", "nl", "pl", "ru", "uk", "zh", "ja", "ko",
            "ar", "tr", "vi", "th", "cs", "ro", "hu", "fi", "da", "sv", "no", "el", "he"
        }

        # Fall back to English if:
        # 1. Confidence is too low
        # 2. Language is not in supported list
        # 3. Language was explicitly specified (trust user input)
        use_lang = detected_lang
        if language is None:  # Only apply fallback logic for auto-detection
            if lang_confidence < min_language_confidence:
                print(f"Warning: Low language confidence ({lang_confidence:.2f}) for '{detected_lang}', falling back to '{fallback_language}'")
                use_lang = fallback_language
            elif detected_lang not in supported_languages:
                print(f"Warning: Detected language '{detected_lang}' may not have alignment support, falling back to '{fallback_language}'")
                use_lang = fallback_language

        # Try to align, with fallback on failure
        try:
            aligned = self.align(result, audio, use_lang)
        except Exception as e:
            if use_lang != fallback_language:
                print(f"Warning: Alignment failed for '{use_lang}': {e}")
                print(f"Retrying with '{fallback_language}'...")
                try:
                    aligned = self.align(result, audio, fallback_language)
                    use_lang = fallback_language
                except Exception as e2:
                    print(f"Warning: Alignment also failed for '{fallback_language}': {e2}")
                    print("Returning unaligned result.")
                    # Return unaligned result as fallback
                    aligned = result
            else:
                print(f"Warning: Alignment failed: {e}")
                print("Returning unaligned result.")
                aligned = result

        return aligned, use_lang


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def segments_to_text(
    segments: list[DiarizedSegment],
    include_timestamps: bool = True,
    include_speakers: bool = True,
) -> str:
    """Convert segments to formatted text.

    Args:
        segments: List of DiarizedSegment objects.
        include_timestamps: Include timestamps in output.
        include_speakers: Include speaker labels in output.

    Returns:
        Formatted transcript text.
    """
    lines = []
    for seg in segments:
        parts = []
        if include_timestamps:
            parts.append(f"[{format_timestamp(seg.start)} - {format_timestamp(seg.end)}]")
        if include_speakers:
            parts.append(f"{seg.speaker_label}:")
        parts.append(seg.text)
        lines.append(" ".join(parts))
    return "\n".join(lines)


def segments_to_srt(segments: list[DiarizedSegment]) -> str:
    """Convert segments to SRT subtitle format.

    Args:
        segments: List of DiarizedSegment objects.

    Returns:
        SRT formatted string.
    """
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_srt_timestamp(seg.start)
        end = format_srt_timestamp(seg.end)
        text = f"{seg.speaker_label}: {seg.text}" if seg.speaker_label else seg.text
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_vtt(segments: list[DiarizedSegment]) -> str:
    """Convert segments to WebVTT subtitle format.

    Args:
        segments: List of DiarizedSegment objects.

    Returns:
        WebVTT formatted string.
    """
    lines = ["WEBVTT", ""]
    for i, seg in enumerate(segments, 1):
        start = format_vtt_timestamp(seg.start)
        end = format_vtt_timestamp(seg.end)
        text = f"{seg.speaker_label}: {seg.text}" if seg.speaker_label else seg.text
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def format_vtt_timestamp(seconds: float) -> str:
    """Format seconds as VTT timestamp (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
