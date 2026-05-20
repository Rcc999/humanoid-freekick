#!/usr/bin/env bash
# Run WHAM on every video in a folder.
# Usage: bash scripts/run_wham_batch.sh <folder_of_videos>
# Example: bash scripts/run_wham_batch.sh freekick-vid

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHAM_DIR="$REPO_ROOT/third_party/WHAM"
source "$REPO_ROOT/.venv/bin/activate"

if [ -z "$1" ]; then
    echo "Usage: bash scripts/run_wham_batch.sh <folder_of_videos>"
    exit 1
fi

# Resolve VIDEO_DIR to absolute path before any cd calls
VIDEO_DIR="$(cd "$1" && pwd)"

cd "$WHAM_DIR"

for VIDEO in "$VIDEO_DIR"/*.{mp4,mov,avi}; do
    # Skip if glob matched nothing
    [ -f "$VIDEO" ] || continue

    NAME="$(basename "$VIDEO" | sed 's/\.[^.]*$//')"
    OUTPUT_DIR="$REPO_ROOT/output/wham/$NAME"

    echo ""
    echo "================================================"
    echo "  Processing: $NAME"
    echo "  Output:     $OUTPUT_DIR"
    echo "================================================"

    mkdir -p "$OUTPUT_DIR"

    python demo.py \
        --video "$VIDEO" \
        --output_pth "$OUTPUT_DIR" \
        --save_pkl \
        --estimate_local_only

    echo "Done: $NAME"
done

echo ""
echo "All done. Outputs in $REPO_ROOT/output/wham/"
echo "Review each visualisation, then run:"
echo "  python scripts/process_wham_output.py --input output/wham/<name>/wham_output.pkl --start X --end Y --output output/<name>_reference_motion.pkl"
