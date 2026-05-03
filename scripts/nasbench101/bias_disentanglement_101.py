"""
Step 5: Bias disentanglement for NAS-Bench-101.

Method:
  1. OLS: regress each proxy on log(param_count); save OLS residuals
  2. Compute both OLS-residual Spearman AND partial-rank Spearman
  3. Decision gate per proxy:
       rho >= 0.30  -> KEEP
       0.10 <= rho < 0.30 -> KEEP_DOCUMENTED
       rho < 0.10   -> EXCLUDE
  Primary metric: partial-rank Spearman (robust to nonlinear size relationship)

Primary output metric:
  partial-rank Spearman rho: rank the proxy, rank the GT, partial out rank(log_params)
  via OLS on ranks, then correlate residual ranks.

Input:  results/nasbench101/transformed_proxy/
        results/nasbench101/audit/gt_accuracies.npy
Output: results/nasbench101/debiased_proxy/
          {proxy}_residuals.npy   shape (N,) float32
          partial_correlations.json
          debiasing_summary.json
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr

ROOT_DIR   = Path("F:/Thesis/Experimentation")
AUDIT_DIR  = ROOT_DIR / "results/nasbench101/audit"
TRANS_DIR  = ROOT_DIR / "results/nasbench101/transformed_proxy"
OUT_DIR    = ROOT_DIR / "results/nasbench101/debiased_proxy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

KEEP_THRESHOLD     = 0.30
DOCUMENT_THRESHOLD = 0.10

PROXIES = [
    ("synflow",  "synflow_log.npy"),
    ("naswot",   "naswot_log.npy"),
    ("zenscore", "zenscore_log.npy"),
]


def ols_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS regression of y on x (with intercept). Returns residuals."""
    X = np.column_stack([np.ones(len(x)), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ coeffs
    return y - pred


def partial_rank_spearman(proxy: np.ndarray, gt: np.ndarray,
                           covariate: np.ndarray) -> float:
    """
    Partial-rank Spearman: correlate rank-residuals of proxy and GT
    after partialling out covariate via OLS on ranks.
    """
    r_proxy = proxy.argsort().argsort().astype(np.float64)
    r_gt    = gt.argsort().argsort().astype(np.float64)
    r_cov   = covariate.argsort().argsort().astype(np.float64)

    res_proxy = ols_residuals(r_proxy, r_cov)
    res_gt    = ols_residuals(r_gt,    r_cov)

    rho, _ = spearmanr(res_proxy, res_gt)
    return float(rho)


def main():
    gt         = np.load(AUDIT_DIR / "gt_accuracies.npy").astype(np.float64)
    log_params = np.load(TRANS_DIR / "param_count_log.npy").astype(np.float64)

    print(f"GT loaded: n={len(gt)}", flush=True)
    print(f"log_params range: [{log_params.min():.3f}, {log_params.max():.3f}]\n", flush=True)

    partial_results  = {}
    debiasing_summary = {}

    for name, fname in PROXIES:
        path = TRANS_DIR / fname
        if not path.exists():
            print(f"  {name}: FILE NOT FOUND -- skipping", flush=True)
            continue

        proxy = np.load(path).astype(np.float64)
        proxy = np.where(np.isfinite(proxy), proxy, np.nanmedian(proxy))

        # OLS residuals (controlling for log_params linearly)
        residuals = ols_residuals(proxy, log_params)

        # OLS-residual Spearman
        rho_ols, _ = spearmanr(residuals, gt)

        # Partial-rank Spearman (primary metric)
        rho_partial = partial_rank_spearman(proxy, gt, log_params)

        # Decision (based on partial-rank rho)
        abs_rho = abs(rho_partial)
        if abs_rho >= KEEP_THRESHOLD:
            decision = "KEEP"
        elif abs_rho >= DOCUMENT_THRESHOLD:
            decision = "KEEP_DOCUMENTED"
        else:
            decision = "EXCLUDE"

        # Save residuals
        np.save(OUT_DIR / f"{name}_residuals.npy", residuals.astype(np.float32))

        partial_results[name] = {
            "ols_residual_rho":   float(rho_ols),
            "partial_rank_rho":   rho_partial,
            "decision":           decision,
        }
        debiasing_summary[name] = {
            "ols_residual_rho":   float(rho_ols),
            "partial_rank_rho":   rho_partial,
            "decision":           decision,
            "residuals_file":     f"{name}_residuals.npy",
        }

        print(f"  {name}:", flush=True)
        print(f"    OLS-residual rho = {rho_ols:.4f}", flush=True)
        print(f"    Partial-rank rho = {rho_partial:.4f}  -> {decision}", flush=True)

    with open(OUT_DIR / "partial_correlations.json", "w") as f:
        json.dump(partial_results, f, indent=2)

    with open(OUT_DIR / "debiasing_summary.json", "w") as f:
        json.dump(debiasing_summary, f, indent=2)

    print(f"\nSaved outputs to {OUT_DIR}", flush=True)

    kept = [n for n, v in partial_results.items() if v["decision"] in ("KEEP", "KEEP_DOCUMENTED")]
    excl = [n for n, v in partial_results.items() if v["decision"] == "EXCLUDE"]
    print(f"\nKept proxies:     {kept}", flush=True)
    print(f"Excluded proxies: {excl}", flush=True)


if __name__ == "__main__":
    main()
