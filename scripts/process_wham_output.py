#!/usr/bin/env python3
"""
Trim, smooth, and normalize WHAM output into a clean reference motion.

Usage:
    python scripts/process_wham_output.py \
        --input output/wham/<clip_name>/<clip_name>/wham_output.pkl \
        --start <frame> \
        --end <frame> \
        --output output/reference_motion.pkl
"""

import argparse
import pickle
import joblib
import numpy as np
from scipy.signal import savgol_filter


def pick_track(data):
    """Pick the track with the most frames (main person)."""
    return max(data.values(), key=lambda t: len(t['frame_ids']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Path to wham_output.pkl")
    parser.add_argument("--start",  type=int, default=0,  help="First frame of run-up")
    parser.add_argument("--end",    type=int, default=-1, help="Frame after ball contact (-1 = all)")
    parser.add_argument("--output", required=True, help="Where to save reference_motion.pkl")
    args = parser.parse_args()

    print("Loading WHAM output...")
    data = joblib.load(args.input)
    track = pick_track(data)

    poses = track['pose']        # (T, 72) SMPL pose params
    trans = track['trans']       # (T, 3)  root translation
    betas = track['betas']       # (T, 10) shape params
    frame_ids = track['frame_ids']

    T = poses.shape[0]
    end = args.end if args.end != -1 else T
    print(f"Total frames: {T}, trimming to {args.start}–{end}")

    poses = poses[args.start:end]
    trans = trans[args.start:end]
    betas = betas[args.start:end]
    n = poses.shape[0]

    # Smooth with Savitzky-Golay (window must be odd and <= n)
    win = min(7, n if n % 2 == 1 else n - 1)
    if win >= 5:
        poses = savgol_filter(poses, window_length=win, polyorder=3, axis=0)
        trans = savgol_filter(trans, window_length=win, polyorder=3, axis=0)

    # Normalize root to origin
    trans -= trans[0]

    result = {
        'poses':    poses,   # (T, 72)
        'trans':    trans,   # (T, 3)
        'betas':    betas,   # (T, 10)
        'n_frames': n,
    }

    with open(args.output, 'wb') as f:
        pickle.dump(result, f)

    print(f"Saved {n} frames to {args.output}")
    print(f"  poses: {poses.shape}")
    print(f"  trans: {trans.shape}")


if __name__ == "__main__":
    main()
