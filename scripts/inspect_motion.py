#!/usr/bin/env python3
"""
Inspect any .npz motion file and print its structure.

Usage:
    python scripts/inspect_motion.py path/to/motion.npz

Works on both AMASS-style and HumanoidSoccer-style .npz files.
"""
import sys
import numpy as np

def inspect(path):
    print(f"\n=== {path} ===")
    d = np.load(path, allow_pickle=True)
    print(f"Keys: {list(d.files)}")
    for k in d.files:
        v = d[k]
        if hasattr(v, 'shape') and v.ndim > 0:
            print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            if v.ndim == 1 and v.shape[0] <= 5:
                print(f"    values: {v}")
            elif v.ndim >= 2:
                print(f"    first row: {v[0]}")
                if k in ['trans', 'root_trans', 'root_pos']:
                    print(f"    Z range: {v[:,2].min():.3f} → {v[:,2].max():.3f}")
        else:
            print(f"  {k}: {v}")

if __name__ == "__main__":
    paths = sys.argv[1:] if len(sys.argv) > 1 else []
    if not paths:
        print("Usage: python scripts/inspect_motion.py path/to/file.npz [...]")
        sys.exit(1)
    for p in paths:
        inspect(p)
