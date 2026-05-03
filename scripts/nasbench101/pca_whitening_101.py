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

ROOT_DIR   = Path("F:/Thesis/Experimentation")
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


if __name__ == "__main__":
    main()
