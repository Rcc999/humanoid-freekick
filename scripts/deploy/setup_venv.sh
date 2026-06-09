#!/usr/bin/env bash
#
# setup_venv.sh — create a Python virtual environment for the hardware
# deployment on the Unitree G1's onboard Jetson.
#
# Usage (run on the Jetson):
#   bash setup_venv.sh
#
# After it finishes:
#   source ~/cr7/.venv/bin/activate
#   python3 deploy_g1_hardware.py --network_interface eth0 --dry_run
#

set -e

VENV_DIR="${VENV_DIR:-$HOME/cr7/.venv}"
REQ_FILE="$(cd "$(dirname "$0")" && pwd)/requirements.txt"

echo "[setup] Creating venv at $VENV_DIR"
python3 -m venv "$VENV_DIR"

echo "[setup] Activating venv"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

echo "[setup] Upgrading pip + wheel"
pip install --upgrade pip wheel

echo "[setup] Installing requirements from $REQ_FILE"
# unitree_sdk2_python isn't on PyPI in every region; if pip install fails,
# fall back to installing from the official GitHub source.
if ! pip install -r "$REQ_FILE"; then
    echo "[setup] pip install from requirements failed — trying GitHub fallback for SDK"
    # Remove the SDK line and install everything else first
    grep -v "^unitree_sdk2_python" "$REQ_FILE" | pip install -r /dev/stdin
    echo "[setup] Installing unitree_sdk2_python from GitHub"
    pip install "git+https://github.com/unitreerobotics/unitree_sdk2_python.git"
fi

echo ""
echo "[setup] DONE."
echo ""
echo "    venv path:  $VENV_DIR"
echo ""
echo "    Activate it whenever you want to run the deployment:"
echo "        source $VENV_DIR/bin/activate"
echo ""
echo "    Verify the install with:"
echo "        python3 -c 'import onnxruntime, onnx, numpy, huggingface_hub, unitree_sdk2py; print(\"OK\")'"
echo ""
