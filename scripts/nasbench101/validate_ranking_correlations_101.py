"""
Step 4: Validate ranking correlations for NAS-Bench-101.

Computes Spearman rho and Kendall tau for each transformed proxy vs GT.
Reports TWO rho values per proxy (as required for thesis):
  1. Global  -- all 423,624 architectures
  2. Competitive -- architectures with GT > 50% only (~380k archs)

Also computes top-5% and top-10% precision (NOT top-1% -- too tight).

Input:  results/nasbench101/transformed_proxy/  +  audit/gt_accuracies.npy
Output: results/nasbench101/proxy_validation/correlation_results.json
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr, kendalltau

ROOT_DIR   = Path(__file__).resolve().parents[2]
AUDIT_DIR  = ROOT_DIR / "results/nasbench101/audit"
TRANS_DIR  = ROOT_DIR / "results/nasbench101/transformed_proxy"
OUT_DIR    = ROOT_DIR / "results/nasbench101/proxy_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COMPETITIVE_GT_THRESHOLD = 50.0   # % accuracy

PROXIES = [
    ("param_count", "param_count_log.npy"),
    ("synflow",     "synflow_log.npy"),
    ("naswot",      "naswot_log.npy"),
    ("zenscore",    "zenscore_log.npy"),
]


def top_k_precision(proxy: np.ndarray, gt: np.ndarray, top_frac: float) -> float:
    """
    Fraction of top-K proxy predictions that are in top-K GT architectures.
    K = int(top_frac * len(gt))
    """
    k = max(1, int(top_frac * len(gt)))
    proxy_top_k = set(np.argsort(proxy)[-k:])
    gt_top_k    = set(np.argsort(gt)[-k:])
    return len(proxy_top_k & gt_top_k) / k


def main():
    gt = np.load(AUDIT_DIR / "gt_accuracies.npy").astype(np.float64)
    comp_mask = gt > COMPETITIVE_GT_THRESHOLD
    n_comp    = comp_mask.sum()
    print(f"GT loaded: n_total={len(gt)}  n_competitive(>{COMPETITIVE_GT_THRESHOLD}%)={n_comp}\n",
          flush=True)

    results = {}

    for name, fname in PROXIES:
        path = TRANS_DIR / fname
        if not path.exists():
            print(f"  {name}: FILE NOT FOUND -- skipping", flush=True)
            continue

        proxy = np.load(path).astype(np.float64)
        # Replace any non-finite values with the median before correlating
        proxy = np.where(np.isfinite(proxy), proxy, np.nanmedian(proxy))

        # Global correlations
        rho_g,  _  = spearmanr(proxy, gt)
        tau_g,  _  = kendalltau(proxy, gt)
        prec5_g    = top_k_precision(proxy, gt, 0.05)
        prec10_g   = top_k_precision(proxy, gt, 0.10)
        prec1_g    = top_k_precision(proxy, gt, 0.01)

        # Competitive correlations (GT > 50%)
        p_comp = proxy[comp_mask]
        g_comp = gt[comp_mask]
        rho_c, _  = spearmanr(p_comp, g_comp)
        tau_c, _  = kendalltau(p_comp, g_comp)
        prec5_c   = top_k_precision(p_comp, g_comp, 0.05)
        prec10_c  = top_k_precision(p_comp, g_comp, 0.10)

        results[name] = {
            "global": {
                "spearman_rho": float(rho_g),
                "kendall_tau":  float(tau_g),
                "top1pct_prec":  float(prec1_g),
                "top5pct_prec":  float(prec5_g),
                "top10pct_prec": float(prec10_g),
            },
            "competitive_gt50": {
                "n":            int(n_comp),
                "spearman_rho": float(rho_c),
                "kendall_tau":  float(tau_c),
                "top5pct_prec":  float(prec5_c),
                "top10pct_prec": float(prec10_c),
            }
        }

        print(f"  {name}:", flush=True)
        print(f"    global      rho={rho_g:.4f}  tau={tau_g:.4f}  "
              f"top5%={prec5_g:.3f}  top10%={prec10_g:.3f}", flush=True)
        print(f"    competitive rho={rho_c:.4f}  tau={tau_c:.4f}  "
              f"top5%={prec5_c:.3f}  top10%={prec10_c:.3f}", flush=True)

    with open(OUT_DIR / "correlation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved correlation_results.json to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
