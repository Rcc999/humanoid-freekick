#!/usr/bin/env python3
"""
Batch-convert all WHAM output clips to AMASS .npz format for ASAP retargeting.

Finds every wham_output.pkl under output/wham/, picks the longest track,
applies smoothing + root normalization, and saves an AMASS .npz per clip.

Usage (mac):
    python scripts/batch_wham_to_amass.py

Outputs go to output/amass/<clip_name>_amass.npz.
Then copy them all to the Spark:
    scp output/amass/*.npz darius@spark-1dc9:~/rayane/humanoid-freekick/third_party/ASAP/humanoidverse/data/motions/raw_tairantestbed_smpl/
"""

import glob
import os
import joblib
import numpy as np
from scipy.signal import savgol_filter


def pick_longest_track(data):
    return max(data.values(), key=lambda t: len(t["frame_ids"]))


def process_track(track, start=0, end=-1):
    poses = track["pose"]       # (T, 72)
    trans = track["trans"]      # (T, 3)
    betas = track["betas"]      # (T, 10)

    T = poses.shape[0]
    end = end if end != -1 else T
    poses = poses[start:end].astype(np.float32)
    trans = trans[start:end].astype(np.float32)
    betas = betas[start:end].astype(np.float32)
    n = poses.shape[0]

    win = min(7, n if n % 2 == 1 else n - 1)
    if win >= 5:
        poses = savgol_filter(poses, window_length=win, polyorder=3, axis=0).astype(np.float32)
        trans = savgol_filter(trans, window_length=win, polyorder=3, axis=0).astype(np.float32)

    trans -= trans[0]
    betas_mean = betas.mean(axis=0)  # (10,)
    return poses, trans, betas_mean, n


def main():
    pkl_paths = glob.glob("output/wham/*/*/wham_output.pkl")
    if not pkl_paths:
        print("No wham_output.pkl files found under output/wham/")
        return

    os.makedirs("output/amass", exist_ok=True)

    for pkl_path in sorted(pkl_paths):
        clip_name = pkl_path.split("/")[2]
        print(f"\n=== {clip_name} ===")

        data = joblib.load(pkl_path)
        track = pick_longest_track(data)
        n_frames = len(track["frame_ids"])
        print(f"  longest track: {n_frames} frames ({n_frames/30:.1f}s @ 30fps)")

        poses, trans, betas, n = process_track(track)

        out_path = f"output/amass/{clip_name}_amass.npz"
        np.savez(
            out_path,
            poses=poses,
            trans=trans,
            betas=betas,
            gender=np.array("neutral", dtype="<U7"),
            mocap_framerate=np.int64(30),
        )
        print(f"  saved → {out_path}  ({n} frames)")

    print(f"\nDone. Files in output/amass/:")
    for f in sorted(glob.glob("output/amass/*.npz")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
