#!/bin/bash

# Transcribe all audio files in a directory
# Usage: ./transcribe_batch.sh <input_dir> [model]

INPUT_DIR="${1:-.}"
MODEL="${2:-gpt-4o-transcribe}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Directory '$INPUT_DIR' not found"
    exit 1
fi

echo "Transcribing files in: $INPUT_DIR"
echo "Using model: $MODEL"
echo

for file in "$INPUT_DIR"/*.{m4a,mp3,mp4,wav,webm}; do
    [ -e "$file" ] || continue
    echo "Processing: $file"
    python3 "$SCRIPT_DIR/transcribe.py" "$file" --model "$MODEL"
    echo
done

echo "Done. Transcripts saved alongside source files."
