"""
Step 3: Analyze proxy distributions for NAS-Bench-101.

Computes per-proxy distribution statistics and saves summary JSON.
Produces no plots (headless server compatible).

Input:  results/nasbench101/transformed_proxy/
Output: results/nasbench101/proxy_distribution/stats_summary.json
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import skew, kurtosis

ROOT_DIR  = Path("F:/Thesis/Experimentation")
TRANS_DIR = ROOT_DIR / "results/nasbench101/transformed_proxy"
OUT_DIR   = ROOT_DIR / "results/nasbench101/proxy_distribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROXIES = [
    ("param_count", "param_count_log.npy"),
    ("synflow",     "synflow_log.npy"),
    ("naswot",      "naswot_log.npy"),
    ("zenscore",    "zenscore_log.npy"),
]


def describe(arr: np.ndarray, name: str) -> dict:
    arr = arr.astype(np.float64)
    finite = arr[np.isfinite(arr)]
    n_finite = len(finite)
    n_inf    = int(np.sum(~np.isfinite(arr)))
    result = {
        "n_total":   int(len(arr)),
        "n_finite":  n_finite,
        "n_nonfinite": n_inf,
        "mean":      float(np.mean(finite)),
        "std":       float(np.std(finite)),
        "min":       float(np.min(finite)),
        "p5":        float(np.percentile(finite, 5)),
        "p25":       float(np.percentile(finite, 25)),
        "median":    float(np.median(finite)),
        "p75":       float(np.percentile(finite, 75)),
        "p95":       float(np.percentile(finite, 95)),
        "max":       float(np.max(finite)),
        "skewness":  float(skew(finite)),
        "kurtosis":  float(kurtosis(finite)),
        "normalized": bool(abs(skew(finite)) <= 2.0),
    }
    flag = "" if result["normalized"] else "  *** NOT NORMALIZED (|skewness| > 2) ***"
    print(f"  {name:15s}: mean={result['mean']:.3f}  std={result['std']:.3f}  "
          f"skew={result['skewness']:.3f}  kurt={result['kurtosis']:.3f}{flag}", flush=True)
    return result


def main():
    summary = {}
    print("Distribution analysis:\n", flush=True)
    for name, fname in PROXIES:
        path = TRANS_DIR / fname
        if not path.exists():
            print(f"  {name}: FILE NOT FOUND -- skipping", flush=True)
            continue
        arr = np.load(path)
        summary[name] = describe(arr, name)

    with open(OUT_DIR / "stats_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved stats_summary.json to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
