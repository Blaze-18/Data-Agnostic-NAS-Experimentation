"""
Step 5: Bias Disentanglement via Partial Correlation

Removes the structural bias introduced by param_count from each proxy, then
measures how much independent signal each proxy retains with GT accuracy.

Method:
  OLS residualisation: regress P ~ param_count_transformed, keep residuals.
  Two partial-rho estimates are computed and cross-checked:
    (1) OLS-residual Spearman: spearmanr(e_P, e_GT)   -- fast, interpretable
    (2) Partial-rank Spearman: spearmanr on rank-residuals -- robust to discrete covariate

Decision thresholds (primary method):
  rho >= 0.30          -> KEEP
  0.10 <= rho < 0.30   -> KEEP (document partial signal)
  rho  < 0.10          -> EXCLUDE

Outputs (all in results/nasbench201/debiased_proxy/):
  naswot_residuals.npy
  zenscore_residuals.npy
  synflow_residuals.npy
  gt_residuals.npy
  phase_a_verification.json   <- Phase A sanity check
  phase_b_residuals.png       <- residual scatter plots
  phase_b_summary.json        <- OLS fit stats per proxy
  partial_correlations.json   <- final decisions
  phase_c_scatter.png         <- residual vs GT-residual scatter
"""

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, Tuple
from scipy.stats import spearmanr
from scipy import stats as scipy_stats

OUT_DIR = Path("results/nasbench201/debiased_proxy")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRANSFORMED_DIR = Path("results/nasbench201/transformed_proxy")
ARCH_DATA_DIR   = Path("data/nasbench201/chunks_clean/arch2infos")

PROXY_FILES = {
    "synflow":     "synflow_transformed.json",
    "naswot":      "naswot_transformed.json",
    "zenscore":    "zenscore_transformed.json",
    "param_count": "param_count_transformed.json",
}

PROXIES_TO_DEBIAS = ["synflow", "naswot", "zenscore"]  # param_count is the covariate

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_transformed_proxy(proxy_name: str) -> Dict[int, float]:
    fpath = TRANSFORMED_DIR / PROXY_FILES[proxy_name]
    with open(fpath) as f:
        data = json.load(f)
    raw = data["results"] if "results" in data else data
    return {int(k): float(v) for k, v in raw.items()}


def load_ground_truth(dataset: str = "cifar10",
                      eval_type: str = "ori-test",
                      epoch: int = 199) -> Dict[int, float]:
    accuracy = {}
    for fpath in sorted(ARCH_DATA_DIR.glob("*.pth")):
        arch_id = int(fpath.stem)
        try:
            data     = torch.load(fpath, weights_only=False)
            arch_data = list(data.values())[0]
            all_res  = arch_data["full"]["all_results"]
            for (dset, seed), res in all_res.items():
                if dset.lower() == dataset.lower():
                    key = f"{eval_type}@{epoch}"
                    if "eval_acc1es" in res and key in res["eval_acc1es"]:
                        accuracy[arch_id] = float(res["eval_acc1es"][key])
                        break
        except Exception:
            continue
    return accuracy


def align_arrays(*dicts: Dict[int, float]) -> Tuple[np.ndarray, ...]:
    """Return aligned numpy arrays for only arch IDs present in ALL dicts."""
    common_ids = sorted(set.intersection(*[set(d.keys()) for d in dicts]))
    arrays = tuple(
        np.array([d[i] for i in common_ids], dtype=np.float64)
        for d in dicts
    )
    return (np.array(common_ids),) + arrays


# ─────────────────────────────────────────────────────────────────────────────
# PHASE A  ─  LOAD & VERIFY
# ─────────────────────────────────────────────────────────────────────────────

def phase_a():
    print("\n" + "=" * 70)
    print("PHASE A: DATA LOADING & VERIFICATION")
    print("=" * 70)

    proxies = {name: load_transformed_proxy(name) for name in PROXY_FILES}
    gt      = load_ground_truth()

    print(f"  GT accuracies loaded:      {len(gt):,}")
    for name, d in proxies.items():
        print(f"  {name:<16} loaded:  {len(d):,}")

    # Align all five dicts together
    all_dicts = [gt] + [proxies[n] for n in ["synflow", "naswot", "zenscore", "param_count"]]
    result    = align_arrays(*all_dicts)
    ids       = result[0]
    gt_arr, synflow_arr, naswot_arr, zenscore_arr, param_arr = result[1:]

    print(f"\n  Aligned architecture count: {len(ids):,}")

    # NaN / Inf check
    arrays_named = {
        "gt":          gt_arr,
        "synflow":     synflow_arr,
        "naswot":      naswot_arr,
        "zenscore":    zenscore_arr,
        "param_count": param_arr,
    }
    any_bad = False
    for name, arr in arrays_named.items():
        n_nan = int(np.sum(~np.isfinite(arr)))
        if n_nan > 0:
            print(f"  WARNING: {name} has {n_nan} non-finite values!")
            any_bad = True
    if not any_bad:
        print("  All arrays: no NaN / Inf values detected.")

    # Basic stats
    print("\n  Quick stats (mean ± std):")
    for name, arr in arrays_named.items():
        print(f"    {name:<16}  mean={arr.mean():.4f}  std={arr.std():.4f}"
              f"  min={arr.min():.4f}  max={arr.max():.4f}")

    # Save verification report
    report = {
        "n_aligned": int(len(ids)),
        "sources": {
            name: {
                "n_loaded": len(proxies[name]) if name != "gt" else len(gt),
                "n_finite": int(np.sum(np.isfinite(arr))),
                "mean": float(arr.mean()),
                "std":  float(arr.std()),
                "min":  float(arr.min()),
                "max":  float(arr.max()),
            }
            for name, arr in arrays_named.items()
        },
        "status": "PASS" if not any_bad else "FAIL",
    }
    out_path = OUT_DIR / "phase_a_verification.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n  Saved: {out_path}")

    return ids, arrays_named


# ─────────────────────────────────────────────────────────────────────────────
# PHASE B  ─  OLS RESIDUALS
# ─────────────────────────────────────────────────────────────────────────────

def ols_residuals(y: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
    """
    Simple OLS: y = b0 + b1*x  → return (residuals, b0, b1, R²).
    """
    X      = np.column_stack([np.ones_like(x), x])
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    b0, b1 = coeffs
    y_hat  = b0 + b1 * x
    resid  = y - y_hat
    ss_res = np.sum(resid ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return resid, float(b0), float(b1), float(r2)


def phase_b(ids: np.ndarray, arrays: Dict[str, np.ndarray]):
    print("\n" + "=" * 70)
    print("PHASE B: OLS RESIDUALISATION")
    print("=" * 70)

    param = arrays["param_count"]
    gt    = arrays["gt"]

    summary = {}
    residuals_out = {}

    # --- GT residuals (regress out param_count) ---
    gt_resid, b0, b1, r2 = ols_residuals(gt, param)
    np.save(OUT_DIR / "gt_residuals.npy", gt_resid)
    residuals_out["gt"] = gt_resid
    summary["gt"] = {"b0": b0, "b1": b1, "r2": r2,
                     "resid_std": float(gt_resid.std())}
    print(f"\n  GT  ~ param_count:  b1={b1:.4f}  R²={r2:.4f}  "
          f"resid_std={gt_resid.std():.4f}")

    # --- Proxy residuals ---
    for name in PROXIES_TO_DEBIAS:
        arr   = arrays[name]
        resid, b0, b1, r2 = ols_residuals(arr, param)
        np.save(OUT_DIR / f"{name}_residuals.npy", resid)
        residuals_out[name] = resid
        summary[name] = {"b0": b0, "b1": b1, "r2": r2,
                         "resid_std": float(resid.std())}
        print(f"  {name:<16} ~ param_count:  b1={b1:.4f}  R²={r2:.4f}  "
              f"resid_std={resid.std():.4f}")

    # --- Save summary JSON ---
    with open(OUT_DIR / "phase_b_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- Residual scatter plots (proxy_residual vs param_count) ---
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle("Phase B — OLS Residuals vs param_count_transformed", fontsize=12)

    plot_targets = [("gt", "GT accuracy")] + [(n, n) for n in PROXIES_TO_DEBIAS]
    for ax, (name, label) in zip(axes, plot_targets):
        resid = residuals_out[name]
        ax.scatter(arrays["param_count"], resid, alpha=0.08, s=4, color="steelblue")
        ax.axhline(0, color="crimson", linewidth=1.0, linestyle="--")
        ax.set_xlabel("param_count (transformed)")
        ax.set_ylabel("Residual")
        ax.set_title(f"{label}\nR²={summary[name]['r2']:.3f}")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "phase_b_residuals.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Saved: {OUT_DIR / 'phase_b_summary.json'}")
    print(f"  Saved: {OUT_DIR / 'phase_b_residuals.png'}")
    for name in list(residuals_out.keys()):
        print(f"  Saved: {OUT_DIR / f'{name}_residuals.npy'}")

    return residuals_out, summary


# ─────────────────────────────────────────────────────────────────────────────
# PHASE C  ─  PARTIAL CORRELATIONS (TWO METHODS)
# ─────────────────────────────────────────────────────────────────────────────

def partial_rank_spearman(y: np.ndarray, x: np.ndarray,
                          cov: np.ndarray) -> Tuple[float, float]:
    """
    Partial Spearman via rank-regression residuals.
    Regresses rank(y) ~ rank(cov) and rank(x) ~ rank(cov), then
    correlates the rank-residuals.  Robust to discrete covariate.
    """
    rank_y   = scipy_stats.rankdata(y)
    rank_x   = scipy_stats.rankdata(x)
    rank_cov = scipy_stats.rankdata(cov)

    r_y,  *_ = ols_residuals(rank_y,  rank_cov)
    r_x,  *_ = ols_residuals(rank_x,  rank_cov)

    rho, pval = spearmanr(r_x, r_y)
    return float(rho), float(pval)


def phase_c(ids: np.ndarray, arrays: Dict[str, np.ndarray],
            residuals: Dict[str, np.ndarray]):
    print("\n" + "=" * 70)
    print("PHASE C: PARTIAL CORRELATION COMPUTATION")
    print("=" * 70)

    gt_resid    = residuals["gt"]
    param       = arrays["param_count"]
    gt          = arrays["gt"]
    results_out = {}

    # Scatter grid: 3 proxies × 2 methods
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle("Phase C — Residualised Proxy vs GT Residuals", fontsize=12)

    for col_idx, name in enumerate(PROXIES_TO_DEBIAS):
        proxy_resid = residuals[name]
        proxy_arr   = arrays[name]

        # Method 1: OLS residual Spearman
        rho_ols,  pval_ols  = spearmanr(proxy_resid, gt_resid)
        rho_ols,  pval_ols  = float(rho_ols), float(pval_ols)

        # Method 2: Partial rank Spearman
        rho_prs, pval_prs = partial_rank_spearman(gt, proxy_arr, param)

        agreement = abs(rho_ols - rho_prs) < 0.05
        primary   = "ols" if agreement else "partial_rank_spearman"
        primary_rho  = rho_ols if agreement else rho_prs
        primary_pval = pval_ols if agreement else pval_prs

        results_out[name] = {
            "ols_rho":          rho_ols,
            "ols_pval":         pval_ols,
            "partial_rank_rho": rho_prs,
            "partial_rank_pval":pval_prs,
            "method_agreement": bool(agreement),
            "primary_method":   primary,
            "partial_rho":      primary_rho,
            "p_value":          primary_pval,
        }

        print(f"\n  {name}")
        print(f"    OLS residual Spearman ρ   = {rho_ols:+.4f}  (p={pval_ols:.2e})")
        print(f"    Partial rank Spearman ρ   = {rho_prs:+.4f}  (p={pval_prs:.2e})")
        print(f"    Methods agree (|Δ|<0.05): {agreement}")
        print(f"    Primary ρ ({primary}): {primary_rho:+.4f}")

        # Row 0: OLS residuals scatter
        ax0 = axes[0, col_idx]
        ax0.scatter(proxy_resid, gt_resid, alpha=0.07, s=4, color="steelblue")
        ax0.set_xlabel(f"{name} residual")
        ax0.set_ylabel("GT residual")
        ax0.set_title(f"{name}\nMethod 1 ρ={rho_ols:+.3f}")

        # Row 1: Rank-residual scatter
        rank_p   = scipy_stats.rankdata(proxy_arr)
        rank_gt  = scipy_stats.rankdata(gt)
        rank_cov = scipy_stats.rankdata(param)
        rr_p,  *_ = ols_residuals(rank_p,  rank_cov)
        rr_gt, *_ = ols_residuals(rank_gt, rank_cov)

        ax1 = axes[1, col_idx]
        ax1.scatter(rr_p, rr_gt, alpha=0.07, s=4, color="darkorange")
        ax1.set_xlabel(f"{name} rank-residual")
        ax1.set_ylabel("GT rank-residual")
        ax1.set_title(f"{name}\nMethod 2 ρ={rho_prs:+.3f}")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "phase_c_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Saved: {OUT_DIR / 'phase_c_scatter.png'}")
    return results_out


# ─────────────────────────────────────────────────────────────────────────────
# PHASE D  ─  DECISION THRESHOLDS
# ─────────────────────────────────────────────────────────────────────────────

def apply_decision(rho: float) -> str:
    if rho >= 0.30:
        return "KEEP"
    elif rho >= 0.10:
        return "KEEP_DOCUMENTED"
    else:
        return "EXCLUDE"


def phase_d(corr_results: Dict):
    print("\n" + "=" * 70)
    print("PHASE D: DECISION THRESHOLDS")
    print("=" * 70)

    print(f"\n  {'Proxy':<16}  {'Partial ρ':>10}  {'p-value':>12}  {'Method':>22}  Decision")
    print("  " + "-" * 76)

    for name in PROXIES_TO_DEBIAS:
        r     = corr_results[name]
        rho   = r["partial_rho"]
        pval  = r["p_value"]
        meth  = r["primary_method"]
        dec   = apply_decision(rho)
        corr_results[name]["decision"] = dec

        flag = " ← EXCLUDED" if dec == "EXCLUDE" else (" ← document" if dec == "KEEP_DOCUMENTED" else "")
        print(f"  {name:<16}  {rho:>+10.4f}  {pval:>12.2e}  {meth:>22}  {dec}{flag}")

    return corr_results


# ─────────────────────────────────────────────────────────────────────────────
# PHASE E  ─  SAVE FINAL JSON + CONSOLE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def phase_e(corr_results: Dict, phase_b_summary: Dict):
    print("\n" + "=" * 70)
    print("PHASE E: SAVING OUTPUTS & FINAL SUMMARY")
    print("=" * 70)

    # Attach global rho for context (from Step 4 results)
    GLOBAL_RHO = {"synflow": 0.164, "naswot": 0.515, "zenscore": 0.554}
    PARAM_CROSS_RHO = {"synflow": 0.23, "naswot": 0.59, "zenscore": 0.62}

    output = {}
    for name in PROXIES_TO_DEBIAS:
        r   = corr_results[name]
        b2  = phase_b_summary[name]
        output[name] = {
            "partial_rho":          r["partial_rho"],
            "p_value":              r["p_value"],
            "primary_method":       r["primary_method"],
            "ols_rho":              r["ols_rho"],
            "partial_rank_rho":     r["partial_rank_rho"],
            "method_agreement":     r["method_agreement"],
            "decision":             r["decision"],
            "param_count_r2":       b2["r2"],        # how much variance param_count explains
            "global_spearman_rho":  GLOBAL_RHO.get(name),
            "param_cross_rho":      PARAM_CROSS_RHO.get(name),
        }

    out_path = OUT_DIR / "partial_correlations.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")

    # Console summary table
    print("\n  ─── STEP 5 RESULTS ─────────────────────────────────────────────────")
    print(f"  {'Proxy':<14}  {'Global ρ':>9}  {'Param R²':>9}  {'Partial ρ':>10}  Decision")
    print("  " + "-" * 62)
    for name in PROXIES_TO_DEBIAS:
        r = output[name]
        print(f"  {name:<14}  {r['global_spearman_rho']:>9.3f}  "
              f"{r['param_count_r2']:>9.3f}  "
              f"{r['partial_rho']:>+10.4f}  "
              f"{r['decision']}")
    print("  " + "-" * 62)
    print("\n  Files written to results/nasbench201/debiased_proxy/:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"    {f.name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ids, arrays   = phase_a()
    residuals, b2 = phase_b(ids, arrays)
    corr_results  = phase_c(ids, arrays, residuals)
    corr_results  = phase_d(corr_results)
    phase_e(corr_results, b2)

    print("\n  Step 5 complete.")
