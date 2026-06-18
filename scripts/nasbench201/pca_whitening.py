"""
Step 6: PCA Whitening of Proxy Features

Takes the three kept proxy metrics (after bias disentanglement) and produces
a whitened feature matrix suitable as input to the surrogate MLP.

Pipeline:
  1. Load & align  — param_count from transformed JSON;
                     naswot and zenscore from OLS residuals (Step 5 output)
  2. Standardise   — z-score each feature (zero mean, unit std)
  3. Fit PCA       — eigen-decompose the 3×3 covariance matrix
  4. Whiten        — divide each PC score by sqrt(eigenvalue) → identity covariance
  5. Validate      — assert col means ≈ 0 and cov(Z) ≈ I_k
  6. Save          — scaler.pkl, pca_model.pkl, arch_ids.npy,
                     whitened_features.npy, pca_summary.json
  7. Plot          — 4-panel diagnostic figure → pca_plots.png

Input features:
  col 0 — param_count_transformed.json    (results/nasbench201/transformed_proxy/)
           Raw log-transformed size signal; the bias covariate from Step 5.
           Enters PCA as an explicit predictor so PC1 can absorb size variance.
  col 1 — naswot_residuals.npy            (results/nasbench201/debiased_proxy/)
           NASWOT score with param_count regressed out via OLS (Step 5).
           By construction: corr(param_count, naswot_residual) ≈ 0.
  col 2 — zenscore_residuals.npy          (results/nasbench201/debiased_proxy/)
           ZenScore with param_count regressed out via OLS (Step 5).
           By construction: corr(param_count, zenscore_residual) ≈ 0.

Pre-PCA input correlations (expected after residualisation):
  param_count  ↔ naswot_residual    ≈  0.000  (forced by OLS)
  param_count  ↔ zenscore_residual  ≈  0.000  (forced by OLS)
  naswot_residual ↔ zenscore_residual ≈ 0.559  (shared activation signal)

Outputs (all in results/nasbench201/pca_whitening/):
  scaler.pkl            — fitted StandardScaler  (for future inference)
  pca_model.pkl         — fitted PCA(whiten=True) (for future inference)
  arch_ids.npy          — (N,)   int32  sorted architecture IDs (row alignment)
  whitened_features.npy — (N, k) float32 whitened principal components
  pca_summary.json      — all metadata: variance ratios, loadings, validation stats
  pca_plots.png         — 4-panel diagnostics

Decision rule for n_retained:
  Retain smallest k such that sum(explained_variance_ratio[:k]) >= VARIANCE_THRESHOLD
  Hard minimum: k >= 2.  With 3 correlated features, k=3 is expected.
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend — no display required
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Column 0: raw log-transformed param_count (size covariate — explicit predictor)
# Column 1: naswot OLS residual after regressing out param_count (Step 5 output)
# Column 2: zenscore OLS residual after regressing out param_count (Step 5 output)
INPUT_FEATURES: List[str] = [
    "param_count",
    "naswot_residual",
    "zenscore_residual",
]

FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "param_count":       "ParamCount (log)",
    "naswot_residual":   "NASWOT (residual)",
    "zenscore_residual": "ZenScore (residual)",
}

TRANSFORMED_DIR = Path("results/nasbench201/transformed_proxy")
DEBIASED_DIR    = Path("results/nasbench201/debiased_proxy")
OUT_DIR         = Path("results/nasbench201/pca_whitening")
OUT_DIR.mkdir(parents=True, exist_ok=True)

VARIANCE_THRESHOLD = 0.99   # retain PCs explaining at least this fraction
MIN_COMPONENTS     = 2      # always keep at least 2 PCs


# ─────────────────────────────────────────────────────────────────────────────
# PHASE A — LOAD & ALIGN
# ─────────────────────────────────────────────────────────────────────────────

def load_features() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the 3-column feature matrix for PCA input:
      col 0 — param_count_transformed  (raw log-transformed score, from JSON)
      col 1 — naswot_residual          (OLS residual from Step 5, .npy)
      col 2 — zenscore_residual        (OLS residual from Step 5, .npy)

    The residual .npy files and the transformed JSON are both ordered by
    sorted(arch_ids), so direct column-stack is safe after a length check.

    Returns
    -------
    X : ndarray, shape (N, 3), float64
        Un-standardised feature matrix.
    arch_ids : ndarray, shape (N,), int32
        Architecture IDs corresponding to each row of X.
    """
    print("\n" + "=" * 70)
    print("PHASE A — LOAD & ALIGN FEATURES")
    print("=" * 70)

    # --- param_count: raw log-transformed score from JSON ---
    fpath = TRANSFORMED_DIR / "param_count_transformed.json"
    if not fpath.exists():
        raise FileNotFoundError(
            f"Not found: {fpath}\nRun transform_proxies.py first."
        )
    with open(fpath) as f:
        data = json.load(f)
    raw = data["results"] if "results" in data else data
    param_dict = {int(k): float(v) for k, v in raw.items()}
    arch_ids = np.array(sorted(param_dict.keys()), dtype=np.int32)
    param_arr = np.array([param_dict[i] for i in arch_ids], dtype=np.float64)
    print(f"  Loaded param_count          : {len(arch_ids):,} architectures  (transformed JSON)")

    # --- naswot and zenscore: OLS residuals from Step 5 ---
    for stem, fname in [("naswot", "naswot_residuals.npy"),
                        ("zenscore", "zenscore_residuals.npy")]:
        rpath = DEBIASED_DIR / fname
        if not rpath.exists():
            raise FileNotFoundError(
                f"Not found: {rpath}\n"
                f"Run bias_disentanglement.py first (Step 5)."
            )
    naswot_r   = np.load(DEBIASED_DIR / "naswot_residuals.npy").astype(np.float64)
    zenscore_r = np.load(DEBIASED_DIR / "zenscore_residuals.npy").astype(np.float64)
    print(f"  Loaded naswot_residual      : {len(naswot_r):,} architectures  (debiased_proxy/)")
    print(f"  Loaded zenscore_residual    : {len(zenscore_r):,} architectures  (debiased_proxy/)")

    # Alignment check: all sources must cover the same architecture set
    n = len(arch_ids)
    if len(naswot_r) != n or len(zenscore_r) != n:
        raise ValueError(
            f"Residual array length mismatch: "
            f"param={n}, naswot_r={len(naswot_r)}, zenscore_r={len(zenscore_r)}\n"
            f"Both bias_disentanglement.py and pca_whitening.py use "
            f"sorted(common_arch_ids) — recheck if either was run on a different subset."
        )

    print(f"\n  Feature matrix: {n:,} architectures × {len(INPUT_FEATURES)} features")
    print(f"  Columns: {INPUT_FEATURES}")

    X = np.column_stack([param_arr, naswot_r, zenscore_r]).astype(np.float64)

    # Sanity: no NaN / Inf
    if np.isnan(X).any() or np.isinf(X).any():
        bad = np.where(~np.isfinite(X))
        raise ValueError(
            f"Non-finite values found in feature matrix at "
            f"{len(bad[0])} positions."
        )
    print("  Finite check: PASS")
    return X, arch_ids


# ─────────────────────────────────────────────────────────────────────────────
# PHASE B — STANDARDISE
# ─────────────────────────────────────────────────────────────────────────────

def standardise(X: np.ndarray) -> Tuple[np.ndarray, StandardScaler]:
    """
    Z-score each feature column.

    Mandatory because the three features have different magnitudes
    (param_count std ≈ 1.06, naswot_residual std ≈ 0.56, zenscore_residual std ≈ 0.52).
    Without this, param_count would dominate PC1 by scale alone.

    Returns
    -------
    X_std   : ndarray, shape (N, F)
    scaler  : fitted StandardScaler (save for inference)
    """
    print("\n" + "=" * 70)
    print("PHASE B — STANDARDISE (z-score per feature)")
    print("=" * 70)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    for i, name in enumerate(INPUT_FEATURES):
        print(
            f"  {name:20s}  mean={scaler.mean_[i]:+.4f}  "
            f"std={scaler.scale_[i]:.4f}  →  "
            f"z-mean={X_std[:, i].mean():+.2e}  z-std={X_std[:, i].std():.4f}"
        )

    return X_std, scaler


# ─────────────────────────────────────────────────────────────────────────────
# PHASE C — FIT PCA
# ─────────────────────────────────────────────────────────────────────────────

def fit_pca(X_std: np.ndarray) -> Tuple[PCA, int]:
    """
    Fit PCA on standardised features and determine how many components to retain.

    sklearn's PCA(whiten=True) sets up the transform but does NOT truncate;
    we still fit all 3 components and decide n_retained separately so the
    summary always reports all eigenvalues.

    Returns
    -------
    pca         : fitted PCA(n_components=F, whiten=True)
    n_retained  : int — number of PCs meeting the variance threshold
    """
    print("\n" + "=" * 70)
    print("PHASE C — FIT PCA")
    print("=" * 70)

    n_features = X_std.shape[1]
    pca = PCA(n_components=n_features, whiten=True, random_state=0)
    pca.fit(X_std)

    evr  = pca.explained_variance_ratio_
    cumr = np.cumsum(evr)

    print(f"\n  {'PC':<5} {'Eigenvalue':>12} {'Var Ratio':>10} {'Cum. Var':>10}")
    print(f"  {'─'*5} {'─'*12} {'─'*10} {'─'*10}")
    for i, (ev, vr, cv) in enumerate(zip(pca.explained_variance_, evr, cumr)):
        print(f"  PC{i+1:<3}  {ev:12.6f}  {vr:10.4%}  {cv:10.4%}")

    # Determine n_retained
    n_retained = int(np.searchsorted(cumr, VARIANCE_THRESHOLD) + 1)
    n_retained = max(n_retained, MIN_COMPONENTS)
    n_retained = min(n_retained, n_features)   # cannot exceed input dims

    print(f"\n  Variance threshold : {VARIANCE_THRESHOLD:.0%}")
    print(f"  Components retained: {n_retained}/{n_features}  "
          f"(cumulative variance = {cumr[n_retained - 1]:.4%})")

    return pca, n_retained


# ─────────────────────────────────────────────────────────────────────────────
# PHASE D — APPLY WHITENING
# ─────────────────────────────────────────────────────────────────────────────

def apply_whitening(X_std: np.ndarray, pca: PCA, n_retained: int) -> np.ndarray:
    """
    Project X_std into whitened PC space and retain the first n_retained PCs.

    sklearn PCA(whiten=True).transform computes:
        Z = (X_std - mean) @ V.T / sqrt(eigenvalue)
    so each output column has unit variance and all columns are uncorrelated.

    Returns
    -------
    Z : ndarray, shape (N, n_retained), float32
    """
    print("\n" + "=" * 70)
    print("PHASE D — APPLY WHITENING")
    print("=" * 70)

    Z_full = pca.transform(X_std)          # (N, n_features); whiten=True applied
    Z = Z_full[:, :n_retained].astype(np.float32)

    print(f"  Input  shape : {X_std.shape}")
    print(f"  Output shape : {Z.shape}  (float32)")
    print(f"\n  Whitened column statistics:")
    print(f"  {'PC':<5} {'Mean':>12} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'─'*5} {'─'*12} {'─'*10} {'─'*10} {'─'*10}")
    for i in range(n_retained):
        col = Z[:, i]
        print(f"  PC{i+1:<3}  {col.mean():+12.6f}  {col.std():10.6f}  "
              f"{col.min():10.4f}  {col.max():10.4f}")

    return Z


# ─────────────────────────────────────────────────────────────────────────────
# PHASE E — VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def validate_whitening(Z: np.ndarray) -> bool:
    """
    Assert:
      1. All column means ≈ 0  (tolerance 1e-5)
      2. Covariance matrix ≈ I (max |deviation| < 1e-4)

    Returns True if both checks pass, False otherwise.
    """
    print("\n" + "=" * 70)
    print("PHASE E — VALIDATE WHITENED OUTPUT")
    print("=" * 70)

    MEAN_TOL = 1e-5
    COV_TOL  = 1e-4

    all_pass = True

    # Check 1: column means
    col_means = Z.mean(axis=0)
    max_mean = np.abs(col_means).max()
    mean_pass = max_mean < MEAN_TOL
    all_pass &= mean_pass
    status = "PASS" if mean_pass else "FAIL"
    print(f"\n  [1] Column means ≈ 0 ... {status}")
    for i, m in enumerate(col_means):
        print(f"      PC{i+1} mean = {m:+.2e}  (tol={MEAN_TOL:.0e})")

    # Check 2: covariance ≈ identity
    cov = np.cov(Z.T.astype(np.float64))
    identity = np.eye(Z.shape[1])
    max_dev = np.abs(cov - identity).max()
    cov_pass = max_dev < COV_TOL
    all_pass &= cov_pass
    status = "PASS" if cov_pass else "FAIL"
    print(f"\n  [2] Cov(Z) ≈ I  max|dev| = {max_dev:.2e}  (tol={COV_TOL:.0e}) ... {status}")
    print(f"\n      Covariance matrix:")
    for row in cov:
        print("      " + "  ".join(f"{v:+.6f}" for v in row))

    if all_pass:
        print("\n  ✓ All validation checks PASSED.")
    else:
        print("\n  ✗ Some validation checks FAILED — review output above.")

    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# PHASE F — SAVE OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(
    scaler: StandardScaler,
    pca: PCA,
    arch_ids: np.ndarray,
    Z: np.ndarray,
    n_retained: int,
    validation_passed: bool,
) -> dict:
    """
    Persist all artifacts to results/nasbench201/pca_whitening/.

    Returns the pca_summary dict (also written as pca_summary.json).
    """
    print("\n" + "=" * 70)
    print("PHASE F — SAVE OUTPUTS")
    print("=" * 70)

    # --- binary artifacts ---
    with open(OUT_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: scaler.pkl")

    with open(OUT_DIR / "pca_model.pkl", "wb") as f:
        pickle.dump(pca, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved: pca_model.pkl")

    np.save(OUT_DIR / "arch_ids.npy", arch_ids)
    print(f"  Saved: arch_ids.npy  ({arch_ids.shape[0]:,} arch IDs)")

    np.save(OUT_DIR / "whitened_features.npy", Z)
    print(f"  Saved: whitened_features.npy  shape={Z.shape}")

    # --- JSON summary ---
    evr  = pca.explained_variance_ratio_.tolist()
    cumr = np.cumsum(pca.explained_variance_ratio_).tolist()

    col_means = Z.mean(axis=0).astype(float).tolist()
    col_stds  = Z.std(axis=0).astype(float).tolist()

    summary = {
        "feature_names":           INPUT_FEATURES,
        "feature_display_names":   FEATURE_DISPLAY_NAMES,
        "n_input_features":        len(INPUT_FEATURES),
        "n_components_retained":   n_retained,
        "n_architectures":         int(arch_ids.shape[0]),
        "variance_threshold_used": VARIANCE_THRESHOLD,
        "explained_variance_ratio":   [round(v, 8) for v in evr],
        "cumulative_variance_ratio":  [round(v, 8) for v in cumr],
        "pca_components":             pca.components_.tolist(),       # (n_components, n_features)
        "pca_explained_variance":     pca.explained_variance_.tolist(),
        "scaler_mean":                scaler.mean_.tolist(),
        "scaler_std":                 scaler.scale_.tolist(),
        "whitened_stats": {
            "col_means": col_means,
            "col_stds":  col_stds,
        },
        "validation_passed": bool(validation_passed),
    }

    summary_path = OUT_DIR / "pca_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: pca_summary.json")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# PHASE G — DIAGNOSTIC PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def plot_diagnostics(
    X: np.ndarray,
    Z: np.ndarray,
    pca: PCA,
    n_retained: int,
) -> None:
    """
    4-panel diagnostic figure:
      [TL] Pearson correlation heatmap — raw input features (shows the problem)
      [TR] Explained variance bar + cumulative line — scree plot
      [BL] PC1 vs PC2 scatter — whitened output distribution
      [BR] Covariance heatmap — whitened output (should be ≈ identity)
    """
    print("\n" + "=" * 70)
    print("PHASE G — DIAGNOSTIC PLOTS")
    print("=" * 70)

    display_names = [FEATURE_DISPLAY_NAMES[n] for n in INPUT_FEATURES]
    pc_labels = [f"PC{i+1}" for i in range(n_retained)]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        "Step 6: PCA Whitening — NAS-Bench-201 Proxy Features",
        fontsize=14, fontweight="bold", y=0.98,
    )

    # ── Panel TL: input feature correlation heatmap ──────────────────────────
    ax = axes[0, 0]
    corr = np.corrcoef(X.T)   # (3, 3) Pearson correlation of raw features
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(len(display_names)))
    ax.set_yticks(range(len(display_names)))
    ax.set_xticklabels(display_names, rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(display_names, fontsize=9)
    for i in range(corr.shape[0]):
        for j in range(corr.shape[1]):
            ax.text(j, i, f"{corr[i, j]:.3f}",
                    ha="center", va="center",
                    color="white" if abs(corr[i, j]) > 0.6 else "black",
                    fontsize=10, fontweight="bold")
    ax.set_title("Input Feature Correlations\n(before PCA)", fontsize=11)

    # ── Panel TR: scree plot ──────────────────────────────────────────────────
    ax = axes[0, 1]
    evr  = pca.explained_variance_ratio_
    cumr = np.cumsum(evr)
    n_all = len(evr)
    x_pos = np.arange(1, n_all + 1)

    bars = ax.bar(x_pos, evr * 100, color="#4C72B0", alpha=0.85, label="Per-PC variance")
    ax2 = ax.twinx()
    ax2.plot(x_pos, cumr * 100, "o-", color="#DD8452", linewidth=2,
             markersize=6, label="Cumulative")
    ax2.axhline(VARIANCE_THRESHOLD * 100, color="#DD8452", linestyle="--",
                linewidth=1, alpha=0.6, label=f"{VARIANCE_THRESHOLD:.0%} threshold")
    ax2.set_ylim(0, 105)
    ax2.set_ylabel("Cumulative variance (%)", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))

    # Annotate retained boundary
    ax.axvline(n_retained + 0.5, color="red", linestyle=":", linewidth=1.5)
    ax.text(n_retained + 0.55, max(evr) * 95, f"Retain {n_retained}",
            color="red", fontsize=9, va="top")

    ax.set_xlabel("Principal Component", fontsize=9)
    ax.set_ylabel("Explained variance (%)", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f%%"))
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"PC{i}" for i in x_pos])

    # Annotate bars
    for bar, val in zip(bars, evr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1%}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=8)
    ax.set_title("Scree Plot — Explained Variance", fontsize=11)

    # ── Panel BL: PC1 vs PC2 scatter ─────────────────────────────────────────
    ax = axes[1, 0]
    # Subsample if N is large (speed up rendering; 15625 is fine but set limit)
    n_pts = min(Z.shape[0], 5000)
    rng   = np.random.default_rng(42)
    idx   = rng.choice(Z.shape[0], size=n_pts, replace=False)
    ax.scatter(Z[idx, 0], Z[idx, 1],
               alpha=0.15, s=4, color="#4C72B0", rasterized=True)
    ax.set_xlabel("PC1 (whitened)", fontsize=9)
    ax.set_ylabel("PC2 (whitened)" if n_retained >= 2 else "PC2 (whitened, excluded)", fontsize=9)
    ax.set_title("Whitened Feature Space\n(PC1 vs PC2)", fontsize=11)
    ax.axhline(0, color="grey", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="grey", linewidth=0.5, linestyle="--")
    if n_pts < Z.shape[0]:
        ax.text(0.97, 0.03, f"n={n_pts:,} sampled", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="grey")
    else:
        ax.text(0.97, 0.03, f"n={Z.shape[0]:,}", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8, color="grey")

    # ── Panel BR: whitened covariance heatmap ────────────────────────────────
    ax = axes[1, 1]
    cov = np.cov(Z.T.astype(np.float64))
    # Ensure 2D even if n_retained == 1
    if cov.ndim == 0:
        cov = cov.reshape(1, 1)

    vmax = max(abs(cov).max(), 1e-6)
    im = ax.imshow(cov, vmin=-vmax, vmax=vmax, cmap="RdBu_r")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n_retained))
    ax.set_yticks(range(n_retained))
    ax.set_xticklabels(pc_labels, fontsize=9)
    ax.set_yticklabels(pc_labels, fontsize=9)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            ax.text(j, i, f"{cov[i, j]:.4f}",
                    ha="center", va="center",
                    color="white" if abs(cov[i, j]) > 0.5 * vmax else "black",
                    fontsize=10, fontweight="bold")
    ax.set_title("Covariance of Whitened Output\n(should be ≈ identity)", fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_path = OUT_DIR / "pca_plots.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: pca_plots.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_pca_whitening(
    transformed_dir: str = "results/nasbench201/transformed_proxy",
    debiased_dir: str = "results/nasbench201/debiased_proxy",
    output_dir: str = "results/nasbench201/pca_whitening",
    variance_threshold: float = 0.99,
) -> None:
    """
    Full PCA whitening pipeline. Entry point.

    Parameters
    ----------
    transformed_dir : path to *_transformed.json files (param_count source)
    debiased_dir    : path to *_residuals.npy files from Step 5
    output_dir      : path to write all outputs
    variance_threshold : cumulative variance fraction to retain
    """
    # Allow callers to override module-level constants at runtime
    global TRANSFORMED_DIR, DEBIASED_DIR, OUT_DIR, VARIANCE_THRESHOLD
    TRANSFORMED_DIR    = Path(transformed_dir)
    DEBIASED_DIR       = Path(debiased_dir)
    OUT_DIR            = Path(output_dir)
    VARIANCE_THRESHOLD = variance_threshold
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "#" * 70)
    print("# STEP 6: PCA WHITENING OF PROXY FEATURES")
    print(f"#   param_count source : {TRANSFORMED_DIR}")
    print(f"#   residuals source   : {DEBIASED_DIR}")
    print(f"#   Output             : {OUT_DIR}")
    print(f"#   Features           : {INPUT_FEATURES}")
    print("#" * 70)

    # Phase A — load & align
    X, arch_ids = load_features()

    # Phase B — standardise
    X_std, scaler = standardise(X)

    # Phase C — fit PCA
    pca, n_retained = fit_pca(X_std)

    # Phase D — whiten
    Z = apply_whitening(X_std, pca, n_retained)

    # Phase E — validate
    validation_passed = validate_whitening(Z)

    # Phase F — save
    summary = save_outputs(scaler, pca, arch_ids, Z, n_retained, validation_passed)

    # Phase G — plot
    plot_diagnostics(X, Z, pca, n_retained)

    # ── Final summary ─────────────────────────────────────────────────────────
    evr  = pca.explained_variance_ratio_
    cumr = np.cumsum(evr)
    print("\n" + "#" * 70)
    print("# STEP 6 COMPLETE")
    print("#" * 70)
    print(f"\n  Input  : {X.shape[0]:,} architectures × {X.shape[1]} features")
    print(f"  Output : {Z.shape[0]:,} architectures × {Z.shape[1]} whitened PCs")
    print()
    print(f"  {'PC':<5}  {'Var %':>8}  {'Cum %':>8}  {'Retained'}")
    print(f"  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*8}")
    for i, (vr, cv) in enumerate(zip(evr, cumr)):
        flag = "✓" if i < n_retained else "—"
        print(f"  PC{i+1:<3}  {vr:8.4%}  {cv:8.4%}  {flag}")
    print()
    print(f"  Validation : {'PASS' if validation_passed else 'FAIL'}")
    print(f"\n  Output files in {OUT_DIR}/")
    for fname in ["scaler.pkl", "pca_model.pkl", "arch_ids.npy",
                  "whitened_features.npy", "pca_summary.json", "pca_plots.png"]:
        fpath = OUT_DIR / fname
        exists = "✓" if fpath.exists() else "✗ MISSING"
        size_str = ""
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            size_str = f"  ({size_kb:.1f} KB)"
        print(f"    {exists}  {fname}{size_str}")


if __name__ == "__main__":
    run_pca_whitening()
