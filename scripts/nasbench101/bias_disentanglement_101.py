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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT_DIR   = Path(__file__).resolve().parents[2]
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

    # Save GT rank-residuals (GT after partialling out log_param_count)
    # Useful for external analysis and reproducing partial-rank scatter plots
    r_gt  = gt.argsort().argsort().astype(np.float64)
    r_cov = log_params.argsort().argsort().astype(np.float64)
    gt_res = ols_residuals(r_gt, r_cov)
    np.save(OUT_DIR / "gt_residuals.npy", gt_res.astype(np.float32))
    print("Saved gt_residuals.npy", flush=True)

    print(f"\nSaved outputs to {OUT_DIR}", flush=True)

    kept = [n for n, v in partial_results.items() if v["decision"] in ("KEEP", "KEEP_DOCUMENTED")]
    excl = [n for n, v in partial_results.items() if v["decision"] == "EXCLUDE"]
    print(f"\nKept proxies:     {kept}", flush=True)
    print(f"Excluded proxies: {excl}", flush=True)

    # ── Plots ───────────────────────────────────────────────────────────────
    proxy_data = {}
    for name, fname in PROXIES:
        path = TRANS_DIR / fname
        if path.exists():
            arr = np.load(path).astype(np.float64)
            proxy_data[name] = np.where(np.isfinite(arr), arr, np.nanmedian(arr))

    n_proxies  = len(proxy_data)
    fig_rows   = 4  # raw-vs-gt | residual-vs-gt | residual-dist | partial-rank-scatter
    fig        = plt.figure(figsize=(6 * n_proxies, 5 * fig_rows))
    gs         = gridspec.GridSpec(fig_rows, n_proxies, figure=fig,
                                   hspace=0.50, wspace=0.35)

    gt_norm = (gt - gt.min()) / (gt.max() - gt.min() + 1e-12)

    for col_idx, (name, _) in enumerate([p for p in PROXIES if p[0] in proxy_data]):
        proxy   = proxy_data[name]
        res     = np.load(OUT_DIR / f"{name}_residuals.npy").astype(np.float64)
        rho_raw, _ = spearmanr(proxy, gt)
        rho_res, _ = spearmanr(res,   gt)
        info    = partial_results[name]
        decision_color = {"KEEP": "#2ecc71", "KEEP_DOCUMENTED": "#f39c12",
                          "EXCLUDE": "#e74c3c"}[info["decision"]]

        # Sample for scatter (max 8000 pts for speed)
        rng   = np.random.default_rng(42)
        idx   = rng.choice(len(gt), min(8000, len(gt)), replace=False)

        # Row 0: Raw proxy vs GT
        ax0 = fig.add_subplot(gs[0, col_idx])
        sc0 = ax0.scatter(proxy[idx], gt[idx], c=gt_norm[idx], cmap='viridis',
                          alpha=0.25, s=4, rasterized=True)
        m0, b0 = np.polyfit(proxy[idx], gt[idx], 1)
        xs = np.linspace(proxy[idx].min(), proxy[idx].max(), 200)
        ax0.plot(xs, m0 * xs + b0, 'r-', lw=1.5, label=f'OLS fit')
        ax0.set_title(f"{name}\nRaw log-proxy vs GT  (ρ={rho_raw:.3f})", fontsize=10)
        ax0.set_xlabel(f"log({name})", fontsize=8)
        ax0.set_ylabel("GT accuracy", fontsize=8)
        ax0.tick_params(labelsize=7)
        plt.colorbar(sc0, ax=ax0, label='GT (norm)', pad=0.01)

        # Row 1: OLS Residuals vs GT
        ax1 = fig.add_subplot(gs[1, col_idx])
        sc1 = ax1.scatter(res[idx], gt[idx], c=gt_norm[idx], cmap='plasma',
                          alpha=0.25, s=4, rasterized=True)
        m1, b1 = np.polyfit(res[idx], gt[idx], 1)
        xs1 = np.linspace(res[idx].min(), res[idx].max(), 200)
        ax1.plot(xs1, m1 * xs1 + b1, 'r-', lw=1.5)
        ax1.axvline(0, color='k', lw=0.8, ls='--', alpha=0.5)
        ax1.set_title(f"Debiased residuals vs GT  (ρ={rho_res:.3f})", fontsize=10)
        ax1.set_xlabel(f"{name} residual", fontsize=8)
        ax1.set_ylabel("GT accuracy", fontsize=8)
        ax1.tick_params(labelsize=7)
        plt.colorbar(sc1, ax=ax1, label='GT (norm)', pad=0.01)

        # Row 2: Residual distribution
        ax2 = fig.add_subplot(gs[2, col_idx])
        ax2.hist(res, bins=120, color=decision_color, alpha=0.75, edgecolor='none')
        ax2.axvline(0, color='k', lw=1, ls='--')
        ax2.set_title(f"Residual distribution\n({info['decision']}  partial ρ={info['partial_rank_rho']:.3f})",
                      fontsize=10, color=decision_color)
        ax2.set_xlabel("Residual value", fontsize=8)
        ax2.set_ylabel("Count", fontsize=8)
        ax2.tick_params(labelsize=7)

        # Row 3: Partial-rank scatter (rank residuals)
        r_proxy = proxy.argsort().argsort().astype(np.float64)
        r_gt    = gt.argsort().argsort().astype(np.float64)
        r_cov   = log_params.argsort().argsort().astype(np.float64)
        res_pr  = ols_residuals(r_proxy, r_cov)
        res_gt  = ols_residuals(r_gt,    r_cov)
        ax3 = fig.add_subplot(gs[3, col_idx])
        ax3.scatter(res_pr[idx], res_gt[idx], c=gt_norm[idx], cmap='coolwarm',
                    alpha=0.20, s=4, rasterized=True)
        m3, b3 = np.polyfit(res_pr[idx], res_gt[idx], 1)
        xs3 = np.linspace(res_pr[idx].min(), res_pr[idx].max(), 200)
        ax3.plot(xs3, m3 * xs3 + b3, 'k-', lw=1.5)
        ax3.set_title(f"Partial-rank scatter  (partial ρ={info['partial_rank_rho']:.3f})", fontsize=10)
        ax3.set_xlabel(f"rank({name}) residual", fontsize=8)
        ax3.set_ylabel("rank(GT) residual", fontsize=8)
        ax3.tick_params(labelsize=7)

    fig.suptitle("NAS-Bench-101 — Step 5: Bias Disentanglement", fontsize=14, fontweight='bold', y=1.005)
    out_png = OUT_DIR / "bias_disentanglement_plots.png"
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"\nSaved plots → {out_png}", flush=True)


if __name__ == "__main__":
    main()
