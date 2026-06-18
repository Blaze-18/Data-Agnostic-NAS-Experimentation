"""
NAS-Bench-101 Feasibility Check
--------------------------------
Uses the nasbench API (with two compatibility patches) to load data and compute:
  - R^2 from log(param_count) -> GT test accuracy  (primary viability check)
  - GT distribution stats (std, range, skewness)
  - Spearman rho of param_count vs GT

Results saved to scripts/nasbench101/feasibility_results.json
Decision threshold: R^2 < 0.40 -> viable for our methodology

Patches applied before running:
  1. envs/nasbench_env/.../nasbench/api.py:
       commented out 'from nasbench.lib import evaluate' (TF1 API removed in TF2)
  2. Set env var PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (proto version mismatch)
"""

import os
import sys

# Must be set BEFORE any tensorflow/protobuf imports
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import json
import numpy as np
from scipy.stats import spearmanr
from scipy import stats as scipy_stats

TFRECORD_PATH = "data/nasbench101/nasbench_full.tfrecord"
OUTPUT_PATH   = "scripts/nasbench101/feasibility_results.json"


def load_data(path):
    """Load param_counts and 108-epoch test accuracies via nasbench API."""
    from nasbench import api

    print("Loading NASBench API (3-8 min for the full 2.1 GB file) ...", flush=True)
    nb = api.NASBench(path)
    print("Loaded. Iterating unique architectures ...", flush=True)

    param_counts  = []
    gt_accuracies = []

    for i, h in enumerate(nb.hash_iterator()):
        if i % 50000 == 0:
            print(f"  {i} / ~423624 ...", flush=True)

        fixed, computed = nb.get_metrics_from_hash(h)
        params = fixed["trainable_parameters"]

        # Average test accuracy over 3 seeds at 108 epochs
        runs = computed[108]  # list of dicts, one per seed
        test_acc_avg = float(np.mean([r["final_test_accuracy"] for r in runs]))

        param_counts.append(params)
        gt_accuracies.append(test_acc_avg)

    return np.array(param_counts, dtype=np.float64), np.array(gt_accuracies, dtype=np.float64)


def compute_feasibility(param_counts, gt_accuracies):
    n = len(param_counts)
    print(f"\nComputing stats on {n} unique architectures ...", flush=True)

    log_pc = np.log(param_counts)

    # OLS: R^2 from log(param_count) -> GT accuracy
    slope, intercept, r, p_val, se = scipy_stats.linregress(log_pc, gt_accuracies)
    r_squared = r ** 2

    # Spearman rho
    rho_pc, _ = spearmanr(log_pc, gt_accuracies)

    # GT distribution
    gt_std  = float(np.std(gt_accuracies))
    gt_mean = float(np.mean(gt_accuracies))
    gt_min  = float(gt_accuracies.min())
    gt_max  = float(gt_accuracies.max())
    gt_skew = float(scipy_stats.skew(gt_accuracies))
    gt_kurt = float(scipy_stats.kurtosis(gt_accuracies))

    # Top-1% spread
    top1pct_n   = max(1, int(0.01 * n))
    top1pct_idx = np.argsort(gt_accuracies)[-top1pct_n:]
    top1pct_std = float(np.std(gt_accuracies[top1pct_idx]))

    viable = r_squared < 0.40

    results = {
        "n_architectures": n,
        "r_squared_param_count_vs_gt": round(r_squared, 4),
        "spearman_rho_param_count_vs_gt": round(float(rho_pc), 4),
        "ols_slope": round(slope, 6),
        "gt_mean_accuracy_pct": round(gt_mean * 100, 4),
        "gt_std_accuracy_pct": round(gt_std * 100, 4),
        "gt_min_accuracy_pct": round(gt_min * 100, 4),
        "gt_max_accuracy_pct": round(gt_max * 100, 4),
        "gt_skewness": round(gt_skew, 4),
        "gt_kurtosis": round(gt_kurt, 4),
        "top1pct_std_accuracy_pct": round(top1pct_std * 100, 4),
        "viable_for_methodology": bool(viable),
        "verdict": (
            "VIABLE -- size explains <40pct of GT variance; proxies have room to add signal."
            if viable else
            "NOT VIABLE -- size dominates GT variance (same structural problem as NATS-Bench SSS)."
        ),
        "threshold_used": "R^2(log_param_count -> GT) < 0.40",
        "reference_nasbench201": "R^2=0.157 (viable)",
        "reference_nats_sss":    "R^2=0.789 (not viable)"
    }
    return results


def main():
    if not os.path.exists(TFRECORD_PATH):
        print(f"ERROR: TFRecord not found at {TFRECORD_PATH}", flush=True)
        sys.exit(1)

    param_counts, gt_accuracies = load_data(TFRECORD_PATH)

    if len(param_counts) == 0:
        print("ERROR: No architectures loaded.", flush=True)
        sys.exit(1)

    results = compute_feasibility(param_counts, gt_accuracies)

    print("\n" + "="*60)
    print("FEASIBILITY CHECK RESULTS")
    print("="*60)
    print(f"  Architectures:               {results['n_architectures']}")
    print(f"  R^2 (log_param -> GT):       {results['r_squared_param_count_vs_gt']}")
    print(f"  Spearman rho (param, GT):    {results['spearman_rho_param_count_vs_gt']}")
    print(f"  GT mean accuracy:            {results['gt_mean_accuracy_pct']}%")
    print(f"  GT std:                      {results['gt_std_accuracy_pct']}%")
    print(f"  GT range:                    {results['gt_min_accuracy_pct']}% -- {results['gt_max_accuracy_pct']}%")
    print(f"  GT skewness:                 {results['gt_skewness']}")
    print(f"  Top-1% std:                  {results['top1pct_std_accuracy_pct']}%")
    print()
    print(f"  VERDICT: {results['verdict']}")
    print("="*60)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
