#!/bin/bash

# Split an audio file into smaller chunks for transcription
# Usage: ./split_audio.sh <input_file> [chunk_duration_seconds] [output_dir]

INPUT_FILE="$1"
CHUNK_DURATION="${2:-600}"  # Default 10 minutes (600 seconds)
OUTPUT_DIR="${3:-}"

if [ -z "$INPUT_FILE" ] || [ ! -f "$INPUT_FILE" ]; then
    echo "Usage: ./split_audio.sh <input_file> [chunk_duration_seconds] [output_dir]"
    echo "  chunk_duration_seconds: default 600 (10 minutes)"
    echo "  output_dir: default <input_basename>-parts"
    exit 1
fi

# Get file info
BASENAME=$(basename "$INPUT_FILE")
FILENAME="${BASENAME%.*}"
EXTENSION="${BASENAME##*.}"
DIRNAME=$(dirname "$INPUT_FILE")

# Set output directory
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="$DIRNAME/${FILENAME}-parts"
fi

# Get total duration
DURATION=$(ffprobe -v quiet -show_entries format=duration -of csv=p=0 "$INPUT_FILE")
DURATION_INT=${DURATION%.*}

echo "Input: $INPUT_FILE"
echo "Duration: ${DURATION_INT}s (~$((DURATION_INT / 60)) minutes)"
echo "Chunk size: ${CHUNK_DURATION}s (~$((CHUNK_DURATION / 60)) minutes)"
echo "Output: $OUTPUT_DIR"
echo

# Calculate number of parts
NUM_PARTS=$(( (DURATION_INT + CHUNK_DURATION - 1) / CHUNK_DURATION ))
echo "Splitting into $NUM_PARTS parts..."
echo

mkdir -p "$OUTPUT_DIR"

for ((i=0; i<NUM_PARTS; i++)); do
    START=$((i * CHUNK_DURATION))
    PART_NUM=$((i + 1))
    OUTPUT_FILE="$OUTPUT_DIR/part${PART_NUM}.${EXTENSION}"

    echo "Creating part $PART_NUM (start: ${START}s)..."
    ffmpeg -y -i "$INPUT_FILE" -ss "$START" -t "$CHUNK_DURATION" -c copy "$OUTPUT_FILE" 2>/dev/null
done

echo
echo "Done. Created $NUM_PARTS parts in: $OUTPUT_DIR"
ls -lh "$OUTPUT_DIR"
