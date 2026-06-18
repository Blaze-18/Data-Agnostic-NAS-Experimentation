"""
Proxy 1 (SSS): Parameter Count
Counts trainable parameters for each SSS architecture using the exact
DynamicShapeTinyNet model (verified against benchmark stored values).

Output: results/nats_bench_sss/raw_proxy_scores/param_count.npy
  Shape (32768,) — param count for each arch index, in benchmark order.
"""
import os
import sys
import pickle
import warnings
import argparse
import numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from proxy_utils_sss import build_sss_model

# ─── Paths ───────────────────────────────────────────────────────────────────
ARCH_DIR    = "data/nats_bench_sss/NATS-sss-v1_0-50262-simple"
OUTPUT_DIR  = "results/nats_bench_sss/raw_proxy_scores"
N_ARCHS     = 32768


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters())


def compute_all(verbose: bool = True) -> np.ndarray:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scores = np.zeros(N_ARCHS, dtype=np.float64)

    for idx in range(N_ARCHS):
        pkl_path = os.path.join(ARCH_DIR, "%d.pickle" % idx)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        arch_str = data["90"]["arch_str"]

        model = build_sss_model(arch_str)
        scores[idx] = float(count_parameters(model))
        del model

        if verbose and (idx + 1) % 500 == 0:
            print("  [param_count] %d / %d  (last arch=%s, params=%.0f)"
                  % (idx + 1, N_ARCHS, arch_str, scores[idx]), flush=True)

    out_path = os.path.join(OUTPUT_DIR, "param_count.npy")
    np.save(out_path, scores)
    print("\nSaved: %s" % out_path, flush=True)
    print("  min=%.0f  max=%.0f  mean=%.0f" %
          (scores.min(), scores.max(), scores.mean()), flush=True)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute param count for all SSS arches")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()
    compute_all(verbose=not args.quiet)
