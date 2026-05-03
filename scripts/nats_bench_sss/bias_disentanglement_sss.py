"""
Step 5 -- Bias Disentanglement via Partial Correlation (NATS-Bench SSS).

Removes the structural capacity bias introduced by param_count from each proxy,
then measures how much independent signal each proxy retains with GT accuracy.

Method:
  OLS residualisation: regress P ~ log(param_count), keep residuals.
  Two partial-rho estimates are computed and cross-checked:
    (1) OLS-residual Spearman: spearmanr(e_P, e_GT)        -- fast, interpretable
    (2) Partial-rank Spearman: spearmanr on rank-residuals  -- primary, robust to discrete

Decision thresholds (primary = partial-rank Spearman):
  rho >= 0.30          -> KEEP
  0.10 <= rho < 0.30   -> KEEP (weak, document)
  rho  < 0.10          -> EXCLUDE

SSS expectation: SSS is a pure capacity search space (only channel widths vary,
topology is fixed).  SynFlow is expected to be excluded (rho near 0 after
debiasing -- nearly identical to param_count).  NASWOT and ZenScore may also
have low partial signal; exclusion is a legitimate finding, not a pipeline error.

Outputs (results/nats_bench_sss/debiased_proxy/):
  gt_residuals.npy
  naswot_residuals.npy
  zenscore_residuals.npy
  synflow_residuals.npy
  phase_a_verification.json
  phase_b_summary.json
  phase_b_residuals.png
  partial_correlations.json    <- primary result: decisions per proxy
  phase_c_scatter.png
"""

import json
import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from scipy.stats import spearmanr
from scipy import stats as scipy_stats

ARCH_DIR    = Path("data/nats_bench_sss/NATS-sss-v1_0-50262-simple")
TRANS_DIR   = Path("results/nats_bench_sss/transformed_proxy")
OUT_DIR     = Path("results/nats_bench_sss/debiased_proxy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ARCHS          = 32_768
PROXIES_TO_DEBIAS = ["synflow", "naswot", "zenscore"]

DECISION_THRESHOLDS = {
    "keep":    0.30,
    "partial": 0.10,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gt_accuracy() -> np.ndarray:
    gt = np.full(N_ARCHS, np.nan)
    print("  Loading GT accuracy ...")
    for idx in range(N_ARCHS):
        if idx % 5000 == 0:
            print(f"    {idx}/{N_ARCHS}", flush=True)
        try:
            with open(ARCH_DIR / f"{idx}.pickle", "rb") as f:
                data = pickle.load(f)
            r = data["90"]["all_results"][("cifar10", 777)]
            gt[idx] = float(r["eval_acc1es"]["ori-test@89"])
        except Exception:
            pass
    return gt


def load_transformed() -> Dict[str, np.ndarray]:
    """Load all log-transformed proxy arrays (indexed 0..32767)."""
    arrays = {}
    for name in ["param_count", "synflow", "naswot", "zenscore"]:
        arrays[name] = np.load(TRANS_DIR / f"{name}.npy")
    return arrays


# ---------------------------------------------------------------------------
# OLS helpers
# ---------------------------------------------------------------------------

def ols_residuals(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """y = b0 + b1*x  ->  (residuals, b0, b1, R^2)."""
    X = np.column_stack([np.ones_like(x), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b0, b1  = coeffs
    y_hat   = b0 + b1 * x
    resid   = y - y_hat
    ss_res  = float(np.sum(resid ** 2))
    ss_tot  = float(np.sum((y - y.mean()) ** 2))
    r2      = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return resid, float(b0), float(b1), float(r2)


def partial_rank_spearman(y: np.ndarray, x: np.ndarray,
                          cov: np.ndarray) -> Tuple[float, float]:
    """
    Partial Spearman via rank-regression residuals.
    Regresses rank(y) ~ rank(cov) and rank(x) ~ rank(cov),
    then correlates the rank-residuals.  Robust to discrete covariate.
    """
    ry  = scipy_stats.rankdata(y)
    rx  = scipy_stats.rankdata(x)
    rc  = scipy_stats.rankdata(cov)
    r_y, *_ = ols_residuals(ry, rc)
    r_x, *_ = ols_residuals(rx, rc)
    rho, pval = spearmanr(r_x, r_y)
    return float(rho), float(pval)


def decide(rho: float) -> str:
    a = abs(rho)
    if a >= DECISION_THRESHOLDS["keep"]:    return "KEEP"
    if a >= DECISION_THRESHOLDS["partial"]: return "KEEP (weak signal)"
    return "EXCLUDE"


# ---------------------------------------------------------------------------
# PHASE A  --  Load & Verify
# ---------------------------------------------------------------------------

def phase_a():
    print("\n" + "=" * 70)
    print("PHASE A: DATA LOADING & VERIFICATION")
    print("=" * 70)

    gt      = load_gt_accuracy()
    proxies = load_transformed()

    # Finite mask -- all arrays must be finite at the same indices
    finite_mask = np.isfinite(gt)
    for arr in proxies.values():
        finite_mask &= np.isfinite(arr)

    ids     = np.where(finite_mask)[0]
    gt_arr  = gt[ids]
    prx_arr = {k: proxies[k][ids] for k in proxies}

    print(f"\n  Aligned: {len(ids):,} / {N_ARCHS:,} architectures")
    for name, arr in prx_arr.items():
        n_nan = int(np.sum(~np.isfinite(arr)))
        print(f"  {name:16}  mean={arr.mean():.4f}  std={arr.std():.4f}"
              f"  nan={n_nan}")

    # Save verification report
    report = {
        "n_aligned": int(len(ids)),
        "n_total":   N_ARCHS,
        "status":    "PASS",
        "arrays": {
            name: {"mean": float(arr.mean()), "std": float(arr.std()),
                   "min": float(arr.min()), "max": float(arr.max())}
            for name, arr in {"gt": gt_arr, **prx_arr}.items()
        },
    }
    out = OUT_DIR / "phase_a_verification.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {out}")

    return ids, gt_arr, prx_arr


# ---------------------------------------------------------------------------
# PHASE B  --  OLS Residuals
# ---------------------------------------------------------------------------

def phase_b(ids: np.ndarray, gt: np.ndarray, proxies: Dict[str, np.ndarray]):
    print("\n" + "=" * 70)
    print("PHASE B: OLS RESIDUALISATION  (covariate = log param_count)")
    print("=" * 70)

    param   = proxies["param_count"]
    summary = {}
    residuals_out: Dict[str, np.ndarray] = {}

    # GT residuals
    gt_resid, b0, b1, r2 = ols_residuals(gt, param)
    np.save(OUT_DIR / "gt_residuals.npy", gt_resid)
    residuals_out["gt"] = gt_resid
    summary["gt"] = {"b0": b0, "b1": b1, "r2": r2,
                     "resid_std": float(gt_resid.std())}
    print(f"\n  GT  ~ param_count:  b1={b1:.4f}  R^2={r2:.4f}  resid_std={gt_resid.std():.4f}")

    # Proxy residuals
    for name in PROXIES_TO_DEBIAS:
        arr = proxies[name]
        resid, b0, b1, r2 = ols_residuals(arr, param)
        np.save(OUT_DIR / f"{name}_residuals.npy", resid)
        residuals_out[name] = resid
        summary[name] = {"b0": b0, "b1": b1, "r2": r2,
                         "resid_std": float(resid.std())}
        print(f"  {name:16} ~ param_count:  b1={b1:.4f}  R^2={r2:.4f}"
              f"  resid_std={resid.std():.4f}")

    with open(OUT_DIR / "phase_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Scatter: residual vs param_count
    plot_names = [("gt", "GT accuracy")] + [(n, n) for n in PROXIES_TO_DEBIAS]
    fig, axes  = plt.subplots(1, len(plot_names), figsize=(5 * len(plot_names), 4))
    fig.suptitle("Phase B -- OLS Residuals vs log(param_count)", fontsize=11)
    for ax, (name, label) in zip(axes, plot_names):
        resid = residuals_out[name]
        ax.scatter(param, resid, alpha=0.06, s=3, color="steelblue", rasterized=True)
        ax.axhline(0, color="crimson", lw=1.0, linestyle="--")
        ax.set_xlabel("log(param_count)")
        ax.set_ylabel("Residual")
        ax.set_title(f"{label}\nR^2={summary[name]['r2']:.3f}")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "phase_b_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {OUT_DIR / 'phase_b_summary.json'}")
    print(f"  Saved: {OUT_DIR / 'phase_b_residuals.png'}")

    return residuals_out, summary


# ---------------------------------------------------------------------------
# PHASE C  --  Partial Correlations
# ---------------------------------------------------------------------------

def phase_c(ids: np.ndarray, gt: np.ndarray,
            proxies: Dict[str, np.ndarray],
            residuals: Dict[str, np.ndarray],
            phase_b_summary: dict):
    print("\n" + "=" * 70)
    print("PHASE C: PARTIAL CORRELATION COMPUTATION")
    print("=" * 70)

    gt_resid = residuals["gt"]
    param    = proxies["param_count"]
    results  = {}

    print(f"\n  {'proxy':16}  {'OLS rho':>8}  {'rank rho':>9}  {'pval':>10}  "
          f"{'param R^2':>10}  decision")
    print("  " + "-" * 75)

    for name in PROXIES_TO_DEBIAS:
        arr      = proxies[name]
        ols_resid = residuals[name]
        r2       = phase_b_summary[name]["r2"]

        # Method 1: OLS-residual Spearman
        rho_ols, pval_ols = spearmanr(ols_resid, gt_resid)

        # Method 2: Partial-rank Spearman (primary)
        rho_prs, pval_prs = partial_rank_spearman(gt, arr, param)

        decision = decide(rho_prs)

        print(f"  {name:16}  {rho_ols:+8.4f}  {rho_prs:+9.4f}  {pval_prs:10.3e}  "
              f"{r2:10.4f}  {decision}")

        results[name] = {
            "ols_rho":    float(rho_ols),
            "ols_pval":   float(pval_ols),
            "prs_rho":    float(rho_prs),   # partial-rank Spearman (primary)
            "prs_pval":   float(pval_prs),
            "param_r2":   float(r2),
            "decision":   decision,
        }

    # Full Spearman (raw, no debiasing) for comparison
    print("\n  Full Spearman rho (no debiasing) for reference:")
    for name in PROXIES_TO_DEBIAS:
        rho_full, _ = spearmanr(proxies[name], gt)
        print(f"  {name:16}  full rho = {rho_full:+.4f}")

    # Save
    with open(OUT_DIR / "partial_correlations.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {OUT_DIR / 'partial_correlations.json'}")

    # Scatter: proxy residual vs GT residual (Phase C plot)
    kept = [n for n in PROXIES_TO_DEBIAS if results[n]["decision"] != "EXCLUDE"]
    n_plot = len(PROXIES_TO_DEBIAS)
    fig, axes = plt.subplots(1, n_plot, figsize=(5 * n_plot, 4))
    if n_plot == 1:
        axes = [axes]
    fig.suptitle("Phase C -- Proxy Residual vs GT Residual", fontsize=11)
    for ax, name in zip(axes, PROXIES_TO_DEBIAS):
        xr = residuals[name]
        yr = residuals["gt"]
        color = "steelblue" if name in kept else "lightgrey"
        ax.scatter(xr, yr, alpha=0.06, s=3, color=color, rasterized=True)
        ax.axhline(0, color="crimson", lw=0.8, linestyle="--")
        ax.axvline(0, color="crimson", lw=0.8, linestyle="--")
        r = results[name]
        ax.set_title(f"{name}\nprs_rho={r['prs_rho']:+.4f}  [{r['decision']}]", fontsize=9)
        ax.set_xlabel(f"{name} residual")
        ax.set_ylabel("GT residual")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "phase_c_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'phase_c_scatter.png'}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Step 5 -- Bias Disentanglement (NATS-Bench SSS)")
    print("=" * 70)
    print(f"\n  Covariate: log(param_count)")
    print(f"  Primary metric: partial-rank Spearman rho")
    print(f"  Decision thresholds: KEEP >= {DECISION_THRESHOLDS['keep']:.2f},"
          f" KEEP-weak >= {DECISION_THRESHOLDS['partial']:.2f}, else EXCLUDE")
    print(f"\n  SSS note: This is a pure capacity search space (fixed topology,")
    print(f"  variable channel widths only).  SynFlow expected to be excluded.")
    print(f"  Exclusion of NASWOT/ZenScore is also a legitimate finding.")

    ids, gt, proxies = phase_a()
    residuals, summary = phase_b(ids, gt, proxies)
    decisions = phase_c(ids, gt, proxies, residuals, summary)

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL DECISIONS")
    print("=" * 70)
    kept = []
    for name, r in decisions.items():
        status = r["decision"]
        print(f"  {name:16}  prs_rho={r['prs_rho']:+.4f}  param_R2={r['param_r2']:.4f}"
              f"  -> {status}")
        if "EXCLUDE" not in status:
            kept.append(name)

    print(f"\n  Proxies retained for PCA: {kept if kept else 'NONE'}")
    if not kept:
        print("\n  PIPELINE NOTE: All activation proxies excluded.")
        print("  SSS pipeline collapses to param_count alone.")
        print("  Document as benchmark characterization finding.")
        print("  Consult supervisor before proceeding to MLP (Step 7).")

    print("\nDone.")


if __name__ == "__main__":
    main()
