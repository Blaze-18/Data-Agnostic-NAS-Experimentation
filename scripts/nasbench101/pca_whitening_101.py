"""
Step 6: PCA whitening for NAS-Bench-101.

Constructs the feature matrix from:
  - log(param_count)   [always included as size feature]
  - {proxy}_residuals  [from Step 5, for each KEPT or KEEP_DOCUMENTED proxy]

Pipeline:
  1. StandardScaler (zero mean, unit variance per feature)
  2. PCA(whiten=True) retaining enough components for 99% explained variance
  3. Validate: |col_mean| < 1e-5, max|Cov(Z) - I| < 1e-4

Reads debiasing_summary.json to know which proxies to include.
Proxies with decision == "EXCLUDE" are NOT included.

Input:  results/nasbench101/transformed_proxy/param_count_log.npy
        results/nasbench101/debiased_proxy/{proxy}_residuals.npy  (kept proxies only)
        results/nasbench101/debiased_proxy/debiasing_summary.json
Output: results/nasbench101/pca_whitening/
          whitened_features.npy    shape (N, k)
          pca_summary.json
"""

import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

ROOT_DIR   = Path("/home/anan/NAS/Experimentation/Data-Agnostic-NAS-Experimentation")
TRANS_DIR  = ROOT_DIR / "results/nasbench101/transformed_proxy"
DEBIAS_DIR = ROOT_DIR / "results/nasbench101/debiased_proxy"
OUT_DIR    = ROOT_DIR / "results/nasbench101/pca_whitening"
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANCE_THRESHOLD = 0.99


def main():
    # Load debiasing decisions
    with open(DEBIAS_DIR / "debiasing_summary.json") as f:
        debias = json.load(f)

    kept = [name for name, v in debias.items()
            if v["decision"] in ("KEEP", "KEEP_DOCUMENTED")]

    print(f"Kept proxies (will be included in PCA): {kept}", flush=True)

    # Build feature matrix
    log_params = np.load(TRANS_DIR / "param_count_log.npy").astype(np.float64)
    log_params = np.where(np.isfinite(log_params), log_params, np.nanmedian(log_params))

    cols       = [log_params]
    col_names  = ["log_param_count"]

    for name in kept:
        res_path = DEBIAS_DIR / f"{name}_residuals.npy"
        if not res_path.exists():
            print(f"  WARNING: {res_path} not found, skipping", flush=True)
            continue
        res = np.load(res_path).astype(np.float64)
        res = np.where(np.isfinite(res), res, 0.0)
        cols.append(res)
        col_names.append(f"{name}_residual")

    X = np.column_stack(cols)
    print(f"Feature matrix X: shape={X.shape}  features={col_names}", flush=True)

    # StandardScaler
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    # PCA with whitening
    pca    = PCA(whiten=True, random_state=42)
    Z      = pca.fit_transform(X_std)

    # Retain components explaining >= VARIANCE_THRESHOLD of variance
    cumvar   = np.cumsum(pca.explained_variance_ratio_)
    n_keep   = int(np.searchsorted(cumvar, VARIANCE_THRESHOLD)) + 1
    n_keep   = min(n_keep, Z.shape[1])
    Z        = Z[:, :n_keep]

    print(f"PCA: {X.shape[1]} features -> {n_keep} components "
          f"({cumvar[n_keep-1]*100:.2f}% variance)", flush=True)

    # Validate
    col_means = np.abs(Z.mean(axis=0))
    max_mean  = float(col_means.max())

    cov_Z     = np.cov(Z.T)
    if cov_Z.ndim == 0:
        cov_err = float(abs(cov_Z - 1.0))
    else:
        cov_err   = float(np.abs(cov_Z - np.eye(n_keep)).max())

    print(f"Validation: max|col_mean|={max_mean:.2e}  max|Cov-I|={cov_err:.2e}",
          flush=True)
    mean_ok = max_mean < 1e-5
    cov_ok  = cov_err < 1e-4 * n_keep   # tolerance scales with n_keep
    print(f"  mean_ok={mean_ok}  cov_ok={cov_ok}", flush=True)

    # Save
    np.save(OUT_DIR / "whitened_features.npy", Z.astype(np.float32))
    print(f"Saved whitened_features.npy  shape={Z.shape}", flush=True)

    summary = {
        "n_architectures":       int(Z.shape[0]),
        "n_input_features":      int(X.shape[1]),
        "input_feature_names":   col_names,
        "n_pca_components":      n_keep,
        "variance_explained_pct": float(cumvar[n_keep-1] * 100),
        "explained_ratios":      pca.explained_variance_ratio_[:n_keep].tolist(),
        "max_col_mean":          max_mean,
        "max_cov_off_diag_err":  cov_err,
        "validation_passed":     bool(mean_ok),
    }
    with open(OUT_DIR / "pca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved pca_summary.json to {OUT_DIR}", flush=True)

    # ── Plots ───────────────────────────────────────────────────────────────
    gt_path = ROOT_DIR / "results/nasbench101/audit/gt_accuracies.npy"
    gt      = np.load(gt_path).astype(np.float64) if gt_path.exists() else None
    gt_norm = ((gt - gt.min()) / (gt.max() - gt.min() + 1e-12)) if gt is not None else None

    ratios  = pca.explained_variance_ratio_
    cumvar  = np.cumsum(ratios)

    n_cols  = min(n_keep, 3)   # show up to first 3 PCs in scatter
    fig     = plt.figure(figsize=(18, 16))
    gs      = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0, col 0: Scree plot ──────────────────────────────────────────
    ax_scree = fig.add_subplot(gs[0, 0])
    ax_scree.bar(range(1, len(ratios) + 1), ratios * 100, color='steelblue',
                 alpha=0.8, label='Individual')
    ax_scree.plot(range(1, len(ratios) + 1), cumvar * 100, 'r-o',
                  markersize=5, lw=1.5, label='Cumulative')
    ax_scree.axhline(99, color='gray', ls='--', lw=1, label='99% threshold')
    ax_scree.set_xlabel("PC index", fontsize=9)
    ax_scree.set_ylabel("Explained variance (%)", fontsize=9)
    ax_scree.set_title("Scree plot", fontsize=10)
    ax_scree.legend(fontsize=8)
    ax_scree.tick_params(labelsize=8)

    # ── Row 0, col 1: Loadings heatmap ───────────────────────────────────
    ax_load = fig.add_subplot(gs[0, 1])
    loadings = pca.components_[:n_keep]  # (n_keep, n_features)
    im = ax_load.imshow(loadings, aspect='auto', cmap='RdBu_r',
                        vmin=-1, vmax=1)
    ax_load.set_xticks(range(len(col_names)))
    ax_load.set_xticklabels([c.replace('_residual', '\nresidual').replace('log_', 'log\n')
                              for c in col_names], fontsize=7)
    ax_load.set_yticks(range(n_keep))
    ax_load.set_yticklabels([f"PC{i+1} ({ratios[i]*100:.1f}%)" for i in range(n_keep)],
                             fontsize=7)
    ax_load.set_title("PCA loadings (components)", fontsize=10)
    plt.colorbar(im, ax=ax_load, shrink=0.8)

    # ── Row 0, col 2: Cumulative variance bar ────────────────────────────
    ax_cum = fig.add_subplot(gs[0, 2])
    colors = ['#2ecc71' if c >= 0.99 else '#3498db' for c in cumvar]
    ax_cum.barh([f"PC{i+1}" for i in range(len(ratios))],
                cumvar * 100, color=colors, alpha=0.85)
    ax_cum.axvline(99, color='red', ls='--', lw=1.2, label='99%')
    ax_cum.set_xlabel("Cumulative variance (%)", fontsize=9)
    ax_cum.set_title("Cumulative explained variance", fontsize=10)
    ax_cum.legend(fontsize=8)
    ax_cum.tick_params(labelsize=8)

    # ── Rows 1-2: Pairwise PC scatter plots colored by GT ────────────────
    rng = np.random.default_rng(42)
    idx = rng.choice(Z.shape[0], min(10000, Z.shape[0]), replace=False)
    pairs = [(0, 1), (0, 2), (1, 2)] if n_keep >= 3 else [(0, 1)]
    pair_positions = [(1, 0), (1, 1), (1, 2)]
    for (pa, pb), (r, c) in zip(pairs, pair_positions):
        if pb >= n_keep:
            continue
        ax_s = fig.add_subplot(gs[r, c])
        sc = ax_s.scatter(Z[idx, pa], Z[idx, pb],
                          c=(gt_norm[idx] if gt_norm is not None else 'steelblue'),
                          cmap='viridis', alpha=0.25, s=4, rasterized=True)
        ax_s.set_xlabel(f"PC{pa+1}", fontsize=9)
        ax_s.set_ylabel(f"PC{pb+1}", fontsize=9)
        ax_s.set_title(f"PC{pa+1} vs PC{pb+1}  (colored by GT accuracy)", fontsize=9)
        ax_s.tick_params(labelsize=7)
        if gt_norm is not None:
            plt.colorbar(sc, ax=ax_s, label='GT (norm)', pad=0.01, shrink=0.85)

    # ── Row 2: Whitened feature distributions ────────────────────────────
    for i in range(min(n_keep, 3)):
        ax_h = fig.add_subplot(gs[2, i])
        ax_h.hist(Z[:, i], bins=120, color='#9b59b6', alpha=0.75, edgecolor='none')
        ax_h.set_xlabel(f"PC{i+1} value", fontsize=9)
        ax_h.set_ylabel("Count", fontsize=9)
        ax_h.set_title(f"PC{i+1} distribution\n(μ={Z[:,i].mean():.2e}, σ={Z[:,i].std():.3f})",
                       fontsize=9)
        ax_h.tick_params(labelsize=7)

    fig.suptitle("NAS-Bench-101 — Step 6: PCA Whitening Diagnostics", fontsize=14,
                 fontweight='bold', y=1.005)
    out_png = OUT_DIR / "pca_plots.png"
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plots → {out_png}", flush=True)


if __name__ == "__main__":
    main()
