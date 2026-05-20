#!/usr/bin/env bash
# Run once on the DGX Spark from the repo root:
#   bash scripts/setup_wham.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHAM_DIR="$REPO_ROOT/third_party/WHAM"

# ── 1. Init submodule if not already done ────────────────────────────────────
git submodule update --init --recursive

# ── 2. Create conda env ──────────────────────────────────────────────────────
conda create -n wham python=3.10 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate wham

# ── 3. PyTorch with CUDA 12.6 (required for GB10 / Blackwell on DGX Spark) ───
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# ── 4. WHAM dependencies ─────────────────────────────────────────────────────
cd "$WHAM_DIR"
pip install -r requirements.txt

# ── 5. Download WHAM pretrained weights ──────────────────────────────────────
bash fetch_demo_data.sh

# ── 6. SMPL body model — MANUAL STEP ─────────────────────────────────────────
echo ""
echo "========================================================"
echo "  MANUAL STEP REQUIRED: SMPL body model"
echo "========================================================"
echo "  1. Register (free) at: https://smpl.is.tue.mpg.de"
echo "  2. Download SMPL_NEUTRAL.pkl"
echo "  3. Place it at:"
echo "     $WHAM_DIR/body_models/smpl/SMPL_NEUTRAL.pkl"
echo "========================================================"
echo ""
echo "Once done, run:  bash scripts/run_wham.sh <path/to/video.mp4>"
