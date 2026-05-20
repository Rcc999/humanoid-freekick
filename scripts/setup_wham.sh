#!/usr/bin/env bash
# Run once on the DGX Spark from the repo root:
#   bash scripts/setup_wham.sh

set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WHAM_DIR="$REPO_ROOT/third_party/WHAM"

# ── 1. Init submodules (no --recursive: avoids pulling vcpkg/Pangolin) ───────
git submodule update --init third_party/WHAM
cd third_party/WHAM
git submodule update --init third-party/ViTPose third-party/DPVO
cd ../..

# ── 2. Python 3.10 via pyenv + venv ─────────────────────────────────────────
pyenv install 3.10.14 --skip-existing
pyenv local 3.10.14
python -m venv .venv
source .venv/bin/activate

# ── 3. PyTorch — aarch64 + CUDA (DGX Spark GB10 / Grace Blackwell) ───────────
# Standard pytorch.org wheels are x86_64 only; use NVIDIA's PyPI index instead
pip install torch torchvision --index-url https://pypi.nvidia.com

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
