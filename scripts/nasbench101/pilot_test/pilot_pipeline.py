"""
Pilot: Steps 2-6 analysis pipeline for NAS-Bench-101 pilot subset.
Runs all analysis after proxy scores are available:
  Step 2: Log-transform proxies
  Step 3: Distribution stats
  Step 4: Ranking correlations (global + competitive)
  Step 5: Bias disentanglement (partial-rank Spearman)
  Step 6: PCA whitening

Input:  results/nasbench101/pilot_test/raw_proxy_scores/
        results/nasbench101/audit/  (first N_PILOT rows)
Output: results/nasbench101/pilot_test/{transformed,distributions,validation,debiased,pca}/

Also copies param_count for the pilot subset from the full audit data.
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr, kendalltau, skew, kurtosis
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

N_PILOT    = 20_000
EPS        = 1e-8
COMP_THRESH = 50.0
KEEP_THRESHOLD     = 0.30
DOCUMENT_THRESHOLD = 0.10

ROOT_DIR   = Path("F:/Thesis/Experimentation")
AUDIT_DIR  = ROOT_DIR / "results/nasbench101/audit"
PILOT_DIR  = ROOT_DIR / "results/nasbench101/pilot_test"
RAW_DIR    = PILOT_DIR / "raw_proxy_scores"

TRANS_DIR  = PILOT_DIR / "transformed_proxy"
DIST_DIR   = PILOT_DIR / "proxy_distribution"
VALID_DIR  = PILOT_DIR / "proxy_validation"
DEBIAS_DIR = PILOT_DIR / "debiased_proxy"
PCA_DIR    = PILOT_DIR / "pca_whitening"

for d in [TRANS_DIR, DIST_DIR, VALID_DIR, DEBIAS_DIR, PCA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log_transform(x, negate=False):
    if negate: x = -x
    return np.log(np.where(x < EPS, EPS, x))

def top_k_precision(proxy, gt, frac):
    k = max(1, int(frac * len(gt)))
    return len(set(np.argsort(proxy)[-k:]) & set(np.argsort(gt)[-k:])) / k

def ols_residuals(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    c, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ c

def partial_rank_spearman(proxy, gt, cov):
    rp = proxy.argsort().argsort().astype(np.float64)
    rg = gt.argsort().argsort().astype(np.float64)
    rc = cov.argsort().argsort().astype(np.float64)
    rho, _ = spearmanr(ols_residuals(rp, rc), ols_residuals(rg, rc))
    return float(rho)

# ---------------------------------------------------------------------------
# Load GT and param_count for pilot subset
# ---------------------------------------------------------------------------

print("=" * 60, flush=True)
print("PILOT PIPELINE  (N={N_PILOT})", flush=True)
print("=" * 60, flush=True)

gt         = np.load(AUDIT_DIR / "gt_accuracies.npy")[:N_PILOT].astype(np.float64)
param_counts = np.load(AUDIT_DIR / "param_counts.npy")[:N_PILOT].astype(np.float64)
np.save(RAW_DIR / "param_count.npy", param_counts)
print(f"GT and param_count loaded: n={len(gt)}", flush=True)

# ---------------------------------------------------------------------------
# STEP 2: Log-transform
# ---------------------------------------------------------------------------
print("\n-- Step 2: Log-transform --", flush=True)

log_pc = log_transform(param_counts)
np.save(TRANS_DIR / "param_count_log.npy", log_pc.astype(np.float32))

transform_summary = {}
for name in ["synflow", "naswot", "zenscore"]:
    raw_path = RAW_DIR / f"{name}.npy"
    if not raw_path.exists():
        print(f"  {name}: NOT FOUND -- skipping", flush=True)
        continue
    raw = np.load(raw_path).astype(np.float64)
    rho_raw, _ = spearmanr(raw, gt)
    negate = bool(rho_raw < 0)
    transformed = log_transform(raw, negate=negate)
    np.save(TRANS_DIR / f"{name}_log.npy", transformed.astype(np.float32))
    rho_t, _ = spearmanr(transformed, gt)
    print(f"  {name}: raw_rho={rho_raw:.4f}  negate={negate}  "
          f"log_rho={rho_t:.4f}", flush=True)
    transform_summary[name] = {"raw_rho": float(rho_raw), "negate": negate,
                                "transformed_rho": float(rho_t)}

with open(TRANS_DIR / "transform_summary.json", "w") as f:
    json.dump(transform_summary, f, indent=2)

# ---------------------------------------------------------------------------
# STEP 3: Distribution stats
# ---------------------------------------------------------------------------
print("\n-- Step 3: Distribution stats --", flush=True)

dist_summary = {}
for name, fname in [("param_count", "param_count_log.npy"),
                     ("synflow",     "synflow_log.npy"),
                     ("naswot",      "naswot_log.npy"),
                     ("zenscore",    "zenscore_log.npy")]:
    path = TRANS_DIR / fname
    if not path.exists(): continue
    arr = np.load(path).astype(np.float64)
    finite = arr[np.isfinite(arr)]
    sk = float(skew(finite))
    dist_summary[name] = {
        "mean": float(np.mean(finite)), "std": float(np.std(finite)),
        "skewness": sk, "kurtosis": float(kurtosis(finite)),
        "normalized": bool(abs(sk) <= 2.0),
    }
    flag = "" if abs(sk) <= 2.0 else "  *** NOT NORMALIZED ***"
    print(f"  {name:15s}: mean={dist_summary[name]['mean']:.3f}  "
          f"std={dist_summary[name]['std']:.3f}  skew={sk:.3f}{flag}", flush=True)

with open(DIST_DIR / "stats_summary.json", "w") as f:
    json.dump(dist_summary, f, indent=2)

# ---------------------------------------------------------------------------
# STEP 4: Ranking correlations
# ---------------------------------------------------------------------------
print("\n-- Step 4: Ranking correlations --", flush=True)

comp_mask = gt > COMP_THRESH
n_comp    = int(comp_mask.sum())
print(f"  Competitive subset (GT>{COMP_THRESH}%): n={n_comp}", flush=True)

corr_results = {}
for name, fname in [("param_count", "param_count_log.npy"),
                     ("synflow",     "synflow_log.npy"),
                     ("naswot",      "naswot_log.npy"),
                     ("zenscore",    "zenscore_log.npy")]:
    path = TRANS_DIR / fname
    if not path.exists(): continue
    proxy = np.load(path).astype(np.float64)
    proxy = np.where(np.isfinite(proxy), proxy, np.nanmedian(proxy))
    rho_g, _ = spearmanr(proxy, gt)
    tau_g, _ = kendalltau(proxy, gt)
    p5g  = top_k_precision(proxy, gt, 0.05)
    p10g = top_k_precision(proxy, gt, 0.10)
    rho_c, _ = spearmanr(proxy[comp_mask], gt[comp_mask])
    tau_c, _ = kendalltau(proxy[comp_mask], gt[comp_mask])
    p5c  = top_k_precision(proxy[comp_mask], gt[comp_mask], 0.05)
    p10c = top_k_precision(proxy[comp_mask], gt[comp_mask], 0.10)
    corr_results[name] = {
        "global":           {"spearman_rho": float(rho_g), "kendall_tau": float(tau_g),
                             "top5pct_prec": float(p5g), "top10pct_prec": float(p10g)},
        "competitive_gt50": {"n": n_comp, "spearman_rho": float(rho_c),
                             "kendall_tau": float(tau_c),
                             "top5pct_prec": float(p5c), "top10pct_prec": float(p10c)},
    }
    print(f"  {name:15s} global rho={rho_g:.4f}  comp rho={rho_c:.4f}  "
          f"top5%={p5g:.3f}", flush=True)

with open(VALID_DIR / "correlation_results.json", "w") as f:
    json.dump(corr_results, f, indent=2)

# ---------------------------------------------------------------------------
# STEP 5: Bias disentanglement
# ---------------------------------------------------------------------------
print("\n-- Step 5: Bias disentanglement --", flush=True)

log_pc_arr = np.load(TRANS_DIR / "param_count_log.npy").astype(np.float64)
debias_summary = {}
kept = []

for name in ["synflow", "naswot", "zenscore"]:
    path = TRANS_DIR / f"{name}_log.npy"
    if not path.exists(): continue
    proxy = np.load(path).astype(np.float64)
    proxy = np.where(np.isfinite(proxy), proxy, np.nanmedian(proxy))
    residuals   = ols_residuals(proxy, log_pc_arr)
    rho_ols, _  = spearmanr(residuals, gt)
    rho_partial = partial_rank_spearman(proxy, gt, log_pc_arr)
    abs_rho = abs(rho_partial)
    if abs_rho >= KEEP_THRESHOLD:
        decision = "KEEP"
    elif abs_rho >= DOCUMENT_THRESHOLD:
        decision = "KEEP_DOCUMENTED"
    else:
        decision = "EXCLUDE"
    np.save(DEBIAS_DIR / f"{name}_residuals.npy", residuals.astype(np.float32))
    debias_summary[name] = {"ols_residual_rho": float(rho_ols),
                             "partial_rank_rho": rho_partial, "decision": decision}
    print(f"  {name:15s} OLS_rho={rho_ols:.4f}  partial_rho={rho_partial:.4f}"
          f"  -> {decision}", flush=True)
    if decision in ("KEEP", "KEEP_DOCUMENTED"):
        kept.append(name)

with open(DEBIAS_DIR / "debiasing_summary.json", "w") as f:
    json.dump(debias_summary, f, indent=2)
print(f"  Kept: {kept}", flush=True)

# ---------------------------------------------------------------------------
# STEP 6: PCA whitening
# ---------------------------------------------------------------------------
print("\n-- Step 6: PCA whitening --", flush=True)

cols      = [log_pc_arr]
col_names = ["log_param_count"]
for name in kept:
    res = np.load(DEBIAS_DIR / f"{name}_residuals.npy").astype(np.float64)
    res = np.where(np.isfinite(res), res, 0.0)
    cols.append(res)
    col_names.append(f"{name}_residual")

X      = np.column_stack(cols)
scaler = StandardScaler()
X_std  = scaler.fit_transform(X)
pca    = PCA(whiten=True, random_state=42)
Z      = pca.fit_transform(X_std)
cumvar = np.cumsum(pca.explained_variance_ratio_)
n_keep = min(int(np.searchsorted(cumvar, 0.99)) + 1, Z.shape[1])
Z      = Z[:, :n_keep]

max_mean = float(np.abs(Z.mean(axis=0)).max())
cov_Z    = np.cov(Z.T) if Z.shape[1] > 1 else np.array([[float(np.var(Z[:, 0]))]])
cov_err  = float(np.abs(cov_Z - np.eye(n_keep)).max())

np.save(PCA_DIR / "whitened_features.npy", Z.astype(np.float32))
pca_summary = {
    "n_architectures": int(N_PILOT),
    "input_feature_names": col_names,
    "n_pca_components": n_keep,
    "variance_explained_pct": float(cumvar[n_keep-1]*100),
    "max_col_mean": max_mean,
    "max_cov_off_diag_err": cov_err,
}
with open(PCA_DIR / "pca_summary.json", "w") as f:
    json.dump(pca_summary, f, indent=2)

print(f"  Features: {col_names}", flush=True)
print(f"  PCA: {X.shape[1]} -> {n_keep} components  "
      f"({cumvar[n_keep-1]*100:.2f}% var)  "
      f"max_mean={max_mean:.2e}  cov_err={cov_err:.2e}", flush=True)

print("\n== Pipeline Steps 2-6 complete ==", flush=True)
print(f"   Outputs in: {PILOT_DIR}", flush=True)
