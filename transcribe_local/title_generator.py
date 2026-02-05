"""Generate descriptive titles for transcripts using OpenAI API."""

import os
from typing import Optional

from .env import load_transcribe_env


def get_first_words(text: str, word_count: int = 8) -> str:
    """Get the first N words from text, cleaned up."""
    words = text.split()[:word_count]
    result = " ".join(words)
    # Add ellipsis if truncated
    if len(text.split()) > word_count:
        result += "..."
    return result


def generate_title(
    transcript_text: str,
    max_chars: int = 2000,
    fallback_word_count: int = 8,
) -> str:
    """Generate a title for a transcript.

    Uses OpenAI API if available, otherwise falls back to first words.

    Args:
        transcript_text: The full transcript text.
        max_chars: Maximum characters to send to API (to limit cost).
        fallback_word_count: Number of words for fallback title.

    Returns:
        Generated title in format: "AI Title - 'first words...'"
        or just "'first words...'" if API unavailable.
    """
    # Get first words for the preview part
    first_words = get_first_words(transcript_text, fallback_word_count)

    # Try to generate AI title
    ai_title = _generate_ai_title(transcript_text[:max_chars])

    if ai_title:
        return f"{ai_title} - '{first_words}'"
    else:
        return f"'{first_words}'"


def _generate_ai_title(text: str) -> Optional[str]:
    """Generate a title using OpenAI API.

    Returns None if API is unavailable or fails.
    """
    load_transcribe_env()
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        return None

    try:
        import openai
        openai.api_key = api_key

        response = openai.chat.completions.create(
            model="gpt-4o-mini",  # Fast and cheap
            messages=[
                {
                    "role": "system",
                    "content": "You generate short, descriptive titles for audio transcripts. "
                    "Respond with ONLY the title, no quotes, no explanation. "
                    "Keep it under 50 characters. Be specific about the topic discussed."
                },
                {
                    "role": "user",
                    "content": f"Generate a title for this transcript:\n\n{text}"
                }
            ],
            max_tokens=50,
            temperature=0.3,
        )

        title = response.choices[0].message.content.strip()
        # Remove quotes if the model added them
        title = title.strip('"\'')
        # Truncate if too long
        if len(title) > 60:
            title = title[:57] + "..."
        return title

    except Exception as e:
        print(f"Warning: Could not generate AI title: {e}")
        return None


def format_transcript_for_title(segments: list) -> str:
    """Format transcript segments into text for title generation.

    Args:
        segments: List of transcript segments (DiarizedSegment or TranscriptSegment).

    Returns:
        Combined text from segments.
    """
    texts = []
    for seg in segments:
        # Handle both DiarizedSegment and TranscriptSegment
        text = getattr(seg, "text", None)
        if text:
            texts.append(text.strip())
    return " ".join(texts)
