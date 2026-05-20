#!/usr/bin/env bash
# Usage: bash scripts/run_wham.sh <path/to/video.mp4>
# Run from the repo root on the DGX Spark.

set -e

VIDEO="$1"
if [ -z "$VIDEO" ]; then
    echo "Usage: bash scripts/run_wham.sh <path/to/video.mp4>"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHAM_DIR="$REPO_ROOT/third_party/WHAM"
OUTPUT_DIR="$REPO_ROOT/output/wham/$(basename "$VIDEO" .mp4)"

source "$REPO_ROOT/.venv/bin/activate"

mkdir -p "$OUTPUT_DIR"

cd "$WHAM_DIR"
python demo.py \
    --video "$VIDEO" \
    --output_dir "$OUTPUT_DIR" \
    --save_pkl \
    --visualize

echo ""
echo "Output saved to: $OUTPUT_DIR"
echo "Next: inspect the output, trim to the run-up + kick segment,"
echo "      then run scripts/process_wham_output.py"
