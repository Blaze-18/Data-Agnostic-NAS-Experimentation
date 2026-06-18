"""
Step 1a: Compute param_count proxy for NAS-Bench-101.
Trivial -- just copies param_counts.npy from Step 0 audit.
No model building required.
"""

import os
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
AUDIT_DIR = ROOT_DIR / "results/nasbench101/audit"
OUT_DIR   = ROOT_DIR / "results/nasbench101/raw_proxy_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    param_counts = np.load(AUDIT_DIR / "param_counts.npy")
    print(f"Loaded param_counts: shape={param_counts.shape}  dtype={param_counts.dtype}")
    print(f"  min={param_counts.min()}  max={param_counts.max()}  mean={param_counts.mean():.0f}")

    out_path = OUT_DIR / "param_count.npy"
    np.save(out_path, param_counts.astype(np.float64))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
