#!/usr/bin/env python3
"""
Convert reference_motion.pkl (WHAM output) to AMASS .npz format
expected by ASAP's fit_smpl_motion.py.

Usage (mac):
    python scripts/pkl_to_amass.py \
        --input output/reference_motion.pkl \
        --output output/cr7_freekick_amass.npz

Then on the Spark copy the .npz to:
    third_party/ASAP/humanoidverse/data/motions/raw_tairantestbed_smpl/
"""

import argparse
import pickle
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="output/reference_motion.pkl")
    parser.add_argument("--output", default="output/cr7_freekick_amass.npz")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        data = pickle.load(f)

    poses = data["poses"].astype(np.float32)   # (T, 72)
    trans = data["trans"].astype(np.float32)   # (T, 3)
    betas = data["betas"].mean(axis=0)         # (10,) average across frames

    np.savez(
        args.output,
        poses=poses,
        trans=trans,
        betas=betas,
        gender=np.array("neutral", dtype="<U7"),
        mocap_framerate=np.int64(30),
    )

    print(f"Saved {poses.shape[0]} frames → {args.output}")
    print(f"  poses: {poses.shape}")
    print(f"  trans: {trans.shape}")
    print(f"  betas: {betas.shape}")


if __name__ == "__main__":
    main()
