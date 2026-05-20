#!/usr/bin/env python3
"""
Trim, smooth, and normalize WHAM output into a clean reference motion.

Usage:
    python scripts/process_wham_output.py \
        --input output/wham/<clip_name>/wham_output.pkl \
        --start <frame> \
        --end <frame> \
        --output output/reference_motion.pkl
"""

import argparse
import pickle
import numpy as np
from scipy.signal import savgol_filter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to wham_output.pkl")
    parser.add_argument("--start",  type=int, default=None, help="First frame of run-up")
    parser.add_argument("--end",    type=int, default=None, help="Frame after ball contact")
    parser.add_argument("--output", required=True, help="Where to save reference_motion.pkl")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    print("Keys:", list(data.keys()))
    print("Total frames:", next(v for v in data.values() if isinstance(v, np.ndarray)).shape[0])

    # Trim
    start = args.start or 0
    end   = args.end   or next(v for v in data.values() if isinstance(v, np.ndarray)).shape[0]
    trimmed = {k: v[start:end] for k, v in data.items() if isinstance(v, np.ndarray)}
    print(f"Trimmed to frames {start}–{end} ({end - start} frames)")

    # Smooth (Savitzky-Golay)
    trimmed["poses"] = savgol_filter(trimmed["poses"], window_length=7, polyorder=3, axis=0)
    trimmed["trans"] = savgol_filter(trimmed["trans"], window_length=7, polyorder=3, axis=0)

    # Normalize root to origin
    trimmed["trans"] -= trimmed["trans"][0]

    with open(args.output, "wb") as f:
        pickle.dump(trimmed, f)

    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
