"""
Step 6 -- PCA Whitening of Proxy Features (NATS-Bench SSS).

Takes param_count (always included as the size covariate) plus any
activation proxies that survived Step 5 bias disentanglement, and produces
a whitened feature matrix suitable as input to the surrogate MLP.

The set of included proxies is determined automatically by reading
results/nats_bench_sss/debiased_proxy/partial_correlations.json -- only
proxies with decision != 'EXCLUDE' are included alongside param_count.

Pipeline:
  A. Load & align  param_count (log-transformed) + kept proxy residuals
  B. Standardise   z-score each feature column
  C. Fit PCA       eigen-decompose the covariance matrix
  D. Whiten        divide each PC score by sqrt(eigenvalue) -> identity cov
  E. Validate      assert col means ~0 and cov(Z) ~I_k
  F. Save          scaler.pkl, pca_model.pkl, arch_ids.npy,
                   whitened_features.npy, pca_summary.json
  G. Plot          diagnostic figure -> pca_plots.png

Outputs (results/nats_bench_sss/pca_whitening/):
  scaler.pkl
  pca_model.pkl
  arch_ids.npy
  whitened_features.npy
  pca_summary.json
  pca_plots.png
"""

import json
import os
import pickle as pkl
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

TRANS_DIR   = Path("results/nats_bench_sss/transformed_proxy")
DEBIASED_DIR= Path("results/nats_bench_sss/debiased_proxy")
OUT_DIR     = Path("results/nats_bench_sss/pca_whitening")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANCE_THRESHOLD = 0.99
MIN_COMPONENTS     = 2


# ---------------------------------------------------------------------------
# PHASE A — Load & Align
# ---------------------------------------------------------------------------

def load_features() -> Tuple[np.ndarray, List[str]]:
    """
    Build the feature matrix:
      col 0  always  param_count (log-transformed)
      col 1+ those activation proxies NOT excluded by Step 5

    Returns (X, feature_names) where X has shape (N, k).
    """
    print("\n" + "=" * 70)
    print("PHASE A -- LOAD & ALIGN FEATURES")
    print("=" * 70)

    # Read Step 5 decisions
    dec_path = DEBIASED_DIR / "partial_correlations.json"
    if not dec_path.exists():
        raise FileNotFoundError(
            f"Not found: {dec_path}\nRun bias_disentanglement_sss.py first."
        )
    with open(dec_path) as f:
        decisions = json.load(f)

    kept = [name for name, r in decisions.items() if "EXCLUDE" not in r["decision"]]
    print(f"\n  Step 5 decisions:")
    for name, r in decisions.items():
        flag = "KEPT" if name in kept else "excluded"
        print(f"    {name:16}  prs_rho={r['prs_rho']:+.4f}  {r['decision']}  [{flag}]")

    if not kept:
        print("\n  WARNING: No activation proxies survived debiasing.")
        print("  PCA will proceed with param_count only (1 feature).")
        print("  This will produce trivial output -- consult supervisor.")

    feature_names = ["param_count"] + [f"{n}_residual" for n in kept]
    print(f"\n  Feature matrix columns: {feature_names}")

    # Load param_count (log-transformed, shape N=32768)
    pc_arr = np.load(TRANS_DIR / "param_count.npy").astype(np.float64)
    n = len(pc_arr)
    print(f"\n  param_count  : {n:,} entries")

    cols = [pc_arr]
    for name in kept:
        rpath = DEBIASED_DIR / f"{name}_residuals.npy"
        if not rpath.exists():
            raise FileNotFoundError(f"Residuals not found: {rpath}")
        arr = np.load(rpath).astype(np.float64)
        if len(arr) != n:
            raise ValueError(f"Length mismatch: param_count={n}, {name}_residuals={len(arr)}")
        cols.append(arr)
        print(f"  {name}_residual: {len(arr):,} entries")

    X = np.column_stack(cols) if len(cols) > 1 else cols[0].reshape(-1, 1)

    if not np.isfinite(X).all():
        raise ValueError("Non-finite values in feature matrix -- check Step 5 outputs.")
    print(f"\n  Feature matrix shape: {X.shape}  finite check: PASS")

    arch_ids = np.arange(n, dtype=np.int32)
    return X, arch_ids, feature_names


# ---------------------------------------------------------------------------
# PHASE B — Standardise
# ---------------------------------------------------------------------------

def standardise(X: np.ndarray, feature_names: List[str]
                ) -> Tuple[np.ndarray, StandardScaler]:
    print("\n" + "=" * 70)
    print("PHASE B -- STANDARDISE (z-score per feature)")
    print("=" * 70)

    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X)

    for i, name in enumerate(feature_names):
        print(f"  {name:24s}  mean={scaler.mean_[i]:+.4f}  "
              f"std={scaler.scale_[i]:.4f}  ->  "
              f"z-mean={X_std[:, i].mean():+.2e}  z-std={X_std[:, i].std():.4f}")

    return X_std, scaler


# ---------------------------------------------------------------------------
# PHASE C — Fit PCA
# ---------------------------------------------------------------------------

def fit_pca(X_std: np.ndarray) -> Tuple[PCA, int]:
    print("\n" + "=" * 70)
    print("PHASE C -- FIT PCA")
    print("=" * 70)

    n_features = X_std.shape[1]
    pca = PCA(n_components=n_features, whiten=True, random_state=0)
    pca.fit(X_std)

    evr  = pca.explained_variance_ratio_
    cumr = np.cumsum(evr)

    print(f"\n  {'PC':<5} {'Eigenvalue':>12} {'Var Ratio':>10} {'Cum. Var':>10}")
    print("  " + "-" * 42)
    for i, (ev, vr, cv) in enumerate(zip(pca.explained_variance_, evr, cumr)):
        print(f"  PC{i+1:<3}  {ev:12.6f}  {vr:10.4%}  {cv:10.4%}")

    n_retained = int(np.searchsorted(cumr, VARIANCE_THRESHOLD) + 1)
    n_retained = max(n_retained, min(MIN_COMPONENTS, n_features))
    n_retained = min(n_retained, n_features)

    print(f"\n  Variance threshold : {VARIANCE_THRESHOLD:.0%}")
    print(f"  Components retained: {n_retained}/{n_features}  "
          f"(cumulative = {cumr[n_retained - 1]:.4%})")

    return pca, n_retained


# ---------------------------------------------------------------------------
# PHASE D — Apply Whitening
# ---------------------------------------------------------------------------

def apply_whitening(X_std: np.ndarray, pca: PCA, n_retained: int) -> np.ndarray:
    print("\n" + "=" * 70)
    print("PHASE D -- APPLY WHITENING")
    print("=" * 70)

    Z_full = pca.transform(X_std)
    Z = Z_full[:, :n_retained].astype(np.float32)

    print(f"  Input  shape : {X_std.shape}")
    print(f"  Output shape : {Z.shape}  (float32)")
    print(f"\n  {'PC':<5} {'Mean':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
    print("  " + "-" * 52)
    for i in range(n_retained):
        col = Z[:, i]
        print(f"  PC{i+1:<3}  {col.mean():+12.6f}  {col.std():10.6f}  "
              f"{col.min():10.4f}  {col.max():10.4f}")

    return Z


# ---------------------------------------------------------------------------
# PHASE E — Validate
# ---------------------------------------------------------------------------

def validate_whitening(Z: np.ndarray) -> bool:
    print("\n" + "=" * 70)
    print("PHASE E -- VALIDATE WHITENED OUTPUT")
    print("=" * 70)

    MEAN_TOL = 1e-4
    COV_TOL  = 1e-3

    all_pass = True

    col_means = Z.mean(axis=0)
    max_mean  = float(np.abs(col_means).max())
    mean_pass = max_mean < MEAN_TOL
    all_pass &= mean_pass
    print(f"\n  [1] Column means ~0  max|mean|={max_mean:.2e}  (tol={MEAN_TOL:.0e}) "
          f"... {'PASS' if mean_pass else 'FAIL'}")

    cov      = np.cov(Z.T.astype(np.float64))
    identity = np.eye(Z.shape[1])
    max_dev  = float(np.abs(cov - identity).max())
    cov_pass = max_dev < COV_TOL
    all_pass &= cov_pass
    print(f"  [2] Cov(Z) ~I  max|dev|={max_dev:.2e}  (tol={COV_TOL:.0e}) "
          f"... {'PASS' if cov_pass else 'FAIL'}")
    print(f"\n  Covariance matrix:")
    for row in (cov if Z.shape[1] > 1 else [[cov.flat[0]]]):
        print("    " + "  ".join(f"{v:+.6f}" for v in row))

    print(f"\n  Validation: {'PASS' if all_pass else 'FAIL'}")
    return all_pass


# ---------------------------------------------------------------------------
# PHASE F — Save
# ---------------------------------------------------------------------------

def save_outputs(scaler: StandardScaler, pca: PCA,
                 arch_ids: np.ndarray, Z: np.ndarray,
                 n_retained: int, feature_names: List[str],
                 validation_passed: bool) -> dict:
    print("\n" + "=" * 70)
    print("PHASE F -- SAVE OUTPUTS")
    print("=" * 70)

    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pkl.dump(scaler, f)
    with open(OUT_DIR / "pca_model.pkl", "wb") as f:
        pkl.dump(pca, f)
    np.save(OUT_DIR / "arch_ids.npy",          arch_ids)
    np.save(OUT_DIR / "whitened_features.npy", Z)

    summary = {
        "n_architectures":     int(len(arch_ids)),
        "n_input_features":    int(pca.n_components_),
        "feature_names":       feature_names,
        "n_retained_pcs":      int(n_retained),
        "variance_threshold":  VARIANCE_THRESHOLD,
        "eigenvalues":         [float(v) for v in pca.explained_variance_],
        "variance_ratios":     [float(v) for v in pca.explained_variance_ratio_],
        "cumulative_variance": list(np.cumsum(pca.explained_variance_ratio_).round(6).tolist()),
        "loadings":            pca.components_.tolist(),
        "scaler_mean":         scaler.mean_.tolist(),
        "scaler_std":          scaler.scale_.tolist(),
        "validation_passed":   bool(validation_passed),
        "whitened_shape":      list(Z.shape),
    }
    with open(OUT_DIR / "pca_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    for name, obj in [
        ("scaler.pkl",            "StandardScaler"),
        ("pca_model.pkl",         "PCA"),
        ("arch_ids.npy",          f"shape {arch_ids.shape}"),
        ("whitened_features.npy", f"shape {Z.shape}"),
        ("pca_summary.json",      "JSON"),
    ]:
        print(f"  Saved: {OUT_DIR / name}  ({obj})")

    return summary


# ---------------------------------------------------------------------------
# PHASE G — Plot
# ---------------------------------------------------------------------------

def plot_diagnostics(X: np.ndarray, X_std: np.ndarray, Z: np.ndarray,
                     pca: PCA, feature_names: List[str], n_retained: int):
    print("\n" + "=" * 70)
    print("PHASE G -- PLOT DIAGNOSTICS")
    print("=" * 70)

    n_feat = X_std.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("NATS-Bench SSS -- PCA Whitening Diagnostics", fontsize=13)

    # Panel 1: Explained variance
    ax = axes[0, 0]
    evr  = pca.explained_variance_ratio_
    cumr = np.cumsum(evr)
    pc_labels = [f"PC{i+1}" for i in range(n_feat)]
    x = np.arange(n_feat)
    ax.bar(x, evr,  alpha=0.7, label="Individual", color="steelblue")
    ax.plot(x, cumr, "ro-", lw=2, label="Cumulative", markersize=6)
    ax.axhline(VARIANCE_THRESHOLD, color="gray", lw=1, linestyle="--",
               label=f"Threshold {VARIANCE_THRESHOLD:.0%}")
    ax.axvline(n_retained - 0.5, color="orange", lw=1.5, linestyle=":",
               label=f"Retain={n_retained}")
    ax.set_xticks(x); ax.set_xticklabels(pc_labels)
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title("Variance Explained")
    ax.legend(fontsize=8)

    # Panel 2: Pre-PCA feature correlation heatmap
    ax = axes[0, 1]
    short = [n.replace("_residual", "_r").replace("param_count", "param") for n in feature_names]
    corr_mat = np.corrcoef(X_std.T)
    im = ax.imshow(corr_mat, cmap="RdYlBu", vmin=-1, vmax=1)
    ax.set_xticks(range(n_feat)); ax.set_xticklabels(short, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(n_feat)); ax.set_yticklabels(short, fontsize=8)
    for i in range(n_feat):
        for j in range(n_feat):
            ax.text(j, i, f"{corr_mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if abs(corr_mat[i, j]) < 0.7 else "white")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Pre-PCA Feature Correlations")

    # Panel 3: Whitened feature covariance
    ax = axes[1, 0]
    Z_shown = Z[:, :min(n_retained, 4)]
    cov     = np.cov(Z_shown.T.astype(np.float64))
    n_show  = Z_shown.shape[1]
    im2 = ax.imshow(cov, cmap="RdYlBu", vmin=-0.01, vmax=1.01)
    pc_ticks = [f"PC{i+1}" for i in range(n_show)]
    ax.set_xticks(range(n_show)); ax.set_xticklabels(pc_ticks)
    ax.set_yticks(range(n_show)); ax.set_yticklabels(pc_ticks)
    for i in range(n_show):
        for j in range(n_show):
            ax.text(j, i, f"{cov[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im2, ax=ax, shrink=0.8)
    ax.set_title("Post-Whitening Covariance (should be ~I)")

    # Panel 4: PC1 vs PC2 scatter (if 2+ PCs retained)
    ax = axes[1, 1]
    if n_retained >= 2:
        ax.scatter(Z[:, 0], Z[:, 1], alpha=0.05, s=3, color="teal", rasterized=True)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.set_title("PC1 vs PC2 (whitened)")
    else:
        ax.hist(Z[:, 0], bins=80, color="teal", alpha=0.8)
        ax.set_xlabel("PC1"); ax.set_ylabel("Count")
        ax.set_title("PC1 distribution (whitened)")

    plt.tight_layout()
    out_png = OUT_DIR / "pca_plots.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Step 6 -- PCA Whitening (NATS-Bench SSS)")
    print("=" * 70)

    X, arch_ids, feature_names = load_features()
    X_std, scaler = standardise(X, feature_names)
    pca, n_retained = fit_pca(X_std)
    Z = apply_whitening(X_std, pca, n_retained)
    valid = validate_whitening(Z)
    summary = save_outputs(scaler, pca, arch_ids, Z, n_retained, feature_names, valid)
    plot_diagnostics(X, X_std, Z, pca, feature_names, n_retained)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Input features  : {feature_names}")
    print(f"  Retained PCs    : {n_retained}")
    print(f"  Output shape    : {Z.shape}")
    print(f"  Validation      : {'PASS' if valid else 'FAIL'}")
    print(f"\n  whitened_features.npy -> use as MLP input in Step 7")
    print("\nDone.")


if __name__ == "__main__":
    main()
