#!env python

import os
import argparse
import openai

from transcribe_local.env import load_transcribe_env

load_transcribe_env()
openai.api_key = os.getenv("OPENAI_API_KEY")


def transcribe_audio(mp3_file, output_file=None, model="gpt-4o-mini-transcribe"):
    if not os.path.isfile(mp3_file):
        raise FileNotFoundError(f"Input file not found: {mp3_file}")

    if output_file is None:
        base, _ = os.path.splitext(mp3_file)
        output_file = f"{base}.txt"

    output_dir = os.path.dirname(output_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    with open(mp3_file, "rb") as audio_fp:
        try:
            result = openai.audio.transcriptions.create(
                model=model,
                file=audio_fp,
            )
        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e

    # Support both dict-like and attribute access depending on SDK version
    text = None
    if hasattr(result, "text"):
        text = result.text
    elif isinstance(result, dict):
        text = result.get("text")

    if not text:
        raise RuntimeError("API returned no text.")

    with open(output_file, "w", encoding="utf-8") as out_fp:
        out_fp.write(text)

    print(f"Saved transcript to: {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe an MP3 file using OpenAI and write text output."
    )
    parser.add_argument("mp3_file", help="Path to the input .mp3 file")
    parser.add_argument(
        "--out", "-o", default=None, help="Path to the output .txt file (optional)"
    )
    parser.add_argument(
        "--model",
        "-m",
        default="gpt-4o-mini-transcribe",
        help="Transcription model to use (e.g., whisper-1)",
    )

    args = parser.parse_args()

    transcribe_audio(args.mp3_file, args.out, args.model)
