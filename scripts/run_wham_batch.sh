#!/usr/bin/env bash
# Run WHAM on every video in a folder.
# Usage: bash scripts/run_wham_batch.sh <folder_of_videos>
# Example: bash scripts/run_wham_batch.sh freekick-vid

set -e

VIDEO_DIR="$1"
if [ -z "$VIDEO_DIR" ]; then
    echo "Usage: bash scripts/run_wham_batch.sh <folder_of_videos>"
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHAM_DIR="$REPO_ROOT/third_party/WHAM"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO_ROOT/.venv/bin/activate"

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

    cd "$WHAM_DIR"
    python demo.py \
        --video "$REPO_ROOT/$VIDEO" \
        --output_pth "$OUTPUT_DIR" \
        --save_pkl \
        --visualize \
        --estimate_local_only

    echo "Done: $NAME"
done

echo ""
echo "All done. Outputs in $REPO_ROOT/output/wham/"
echo "Review each visualisation, then run:"
echo "  python scripts/process_wham_output.py --input output/wham/<name>/wham_output.pkl --start X --end Y --output output/<name>_reference_motion.pkl"
