"""
Step 7: Surrogate MLP for NAS-Bench-101.

Architecture: k -> 64 -> 32 -> 1  (k = number of whitened PCA features)
Loss:         Pairwise ranking loss (not MSE) to preserve rank order
Split:        80% train / 10% val / 10% test  (fixed seed=42)
Optimizer:    Adam lr=1e-3, early stopping on val loss (patience=20)

Ablation: trains 4 model variants and records all metrics:
  1. size_only      -- param_count_log only (no debiasing, no PCA)
  2. best_raw       -- best single raw proxy (highest global rho from Step 4)
  3. pca_raw        -- PCA on all raw transformed proxies (no debiasing)
  4. full_pipeline  -- whitened debiased features from Step 6 (proposed method)

Evaluation:
  Spearman rho (global and competitive GT>50%)
  Kendall tau
  Top-5% and top-10% precision
  Top-1% precision (reported but not primary metric)

Input:  results/nasbench101/pca_whitening/whitened_features.npy
        results/nasbench101/transformed_proxy/
        results/nasbench101/audit/gt_accuracies.npy
        results/nasbench101/proxy_validation/correlation_results.json
Output: results/nasbench101/surrogate_mlp/
          mlp_results.json
          ablation_table.json
"""

import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from scipy.stats import spearmanr, kendalltau
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

ROOT_DIR    = Path("F:/Thesis/Experimentation")
AUDIT_DIR   = ROOT_DIR / "results/nasbench101/audit"
TRANS_DIR   = ROOT_DIR / "results/nasbench101/transformed_proxy"
VALID_DIR   = ROOT_DIR / "results/nasbench101/proxy_validation"
PCA_DIR     = ROOT_DIR / "results/nasbench101/pca_whitening"
OUT_DIR     = ROOT_DIR / "results/nasbench101/surrogate_mlp"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
TRAIN_FRAC  = 0.80
VAL_FRAC    = 0.10
LR          = 1e-3
PATIENCE    = 20
MAX_EPOCHS  = 500
BATCH_SIZE  = 2048

COMP_THRESH = 50.0   # % GT for competitive subset


# ---------------------------------------------------------------------------
# MLP definition
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Pairwise ranking loss (margin = 0)
# ---------------------------------------------------------------------------

def pairwise_ranking_loss(pred: torch.Tensor, target: torch.Tensor,
                          n_pairs: int = 2048) -> torch.Tensor:
    """
    Sample random pairs (i, j) where target[i] > target[j].
    Loss = mean(ReLU(-(pred[i] - pred[j]))).
    """
    idx_i = torch.randint(0, len(pred), (n_pairs,))
    idx_j = torch.randint(0, len(pred), (n_pairs,))
    mask  = target[idx_i] > target[idx_j]
    if mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)
    diff = pred[idx_i[mask]] - pred[idx_j[mask]]
    return torch.relu(-diff).mean()


# ---------------------------------------------------------------------------
# Train + evaluate one MLP variant
# ---------------------------------------------------------------------------

def train_and_eval(X_all: np.ndarray, gt_all: np.ndarray,
                   variant_name: str) -> dict:
    """
    Train an MLP on X_all -> gt_all using pairwise ranking loss.
    Returns dict of evaluation metrics.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n = len(gt_all)
    idx = np.random.permutation(n)
    n_train = int(TRAIN_FRAC * n)
    n_val   = int(VAL_FRAC   * n)

    idx_train = idx[:n_train]
    idx_val   = idx[n_train:n_train + n_val]
    idx_test  = idx[n_train + n_val:]

    X_tr  = torch.from_numpy(X_all[idx_train].astype(np.float32))
    y_tr  = torch.from_numpy(gt_all[idx_train].astype(np.float32))
    X_val = torch.from_numpy(X_all[idx_val].astype(np.float32))
    y_val = torch.from_numpy(gt_all[idx_val].astype(np.float32))
    X_te  = torch.from_numpy(X_all[idx_test].astype(np.float32))
    y_te  = gt_all[idx_test]

    model     = MLP(in_dim=X_all.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        # Mini-batch loop
        perm   = torch.randperm(len(X_tr))
        losses = []
        for start in range(0, len(X_tr), BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            xb, yb    = X_tr[batch_idx], y_tr[batch_idx]
            pred      = model(xb)
            loss      = pairwise_ranking_loss(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = pairwise_ranking_loss(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"    [{variant_name}] Early stop epoch={epoch+1}  "
                  f"val_loss={best_val_loss:.4f}", flush=True)
            break

    # Restore best weights
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_pred = model(X_te).numpy()

    # Metrics on test set
    comp_mask = y_te > COMP_THRESH

    rho_g, _  = spearmanr(test_pred, y_te)
    tau_g, _  = kendalltau(test_pred, y_te)

    rho_c = float("nan")
    tau_c = float("nan")
    if comp_mask.sum() >= 10:
        rho_c, _ = spearmanr(test_pred[comp_mask], y_te[comp_mask])
        tau_c, _ = kendalltau(test_pred[comp_mask], y_te[comp_mask])

    def topk_prec(pred, gt, frac):
        k = max(1, int(frac * len(gt)))
        return len(set(np.argsort(pred)[-k:]) & set(np.argsort(gt)[-k:])) / k

    prec1  = topk_prec(test_pred, y_te, 0.01)
    prec5  = topk_prec(test_pred, y_te, 0.05)
    prec10 = topk_prec(test_pred, y_te, 0.10)

    metrics = {
        "variant":            variant_name,
        "n_test":             int(len(y_te)),
        "n_features":         int(X_all.shape[1]),
        "spearman_rho_global":      float(rho_g),
        "spearman_rho_competitive": float(rho_c),
        "kendall_tau_global":       float(tau_g),
        "kendall_tau_competitive":  float(tau_c),
        "top1pct_precision":        float(prec1),
        "top5pct_precision":        float(prec5),
        "top10pct_precision":       float(prec10),
        "best_val_loss":            float(best_val_loss),
    }

    print(f"    [{variant_name}] rho_global={rho_g:.4f}  rho_comp={rho_c:.4f}  "
          f"top5%={prec5:.3f}  top10%={prec10:.3f}", flush=True)

    return metrics


# ---------------------------------------------------------------------------
# Build feature matrices for each ablation variant
# ---------------------------------------------------------------------------

def build_size_only(gt):
    """Variant 1: log(param_count) only."""
    log_pc = np.load(TRANS_DIR / "param_count_log.npy").astype(np.float64)
    log_pc = np.where(np.isfinite(log_pc), log_pc, np.nanmedian(log_pc))
    scaler = StandardScaler()
    X = scaler.fit_transform(log_pc.reshape(-1, 1))
    return X


def build_best_raw(gt):
    """Variant 2: best single raw proxy (highest global Spearman rho, from Step 4)."""
    with open(VALID_DIR / "correlation_results.json") as f:
        corr = json.load(f)

    proxies = ["synflow", "naswot", "zenscore"]
    best_name = max(proxies,
                    key=lambda p: abs(corr.get(p, {}).get("global", {})
                                      .get("spearman_rho", 0.0)))
    print(f"  Best raw proxy: {best_name}  rho="
          f"{corr[best_name]['global']['spearman_rho']:.4f}", flush=True)

    path = TRANS_DIR / f"{best_name}_log.npy"
    if not path.exists():
        # Fallback to param_count
        path = TRANS_DIR / "param_count_log.npy"
    arr = np.load(path).astype(np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nanmedian(arr))
    scaler = StandardScaler()
    X = scaler.fit_transform(arr.reshape(-1, 1))
    return X


def build_pca_raw(gt):
    """Variant 3: PCA on all raw transformed proxies (no debiasing)."""
    cols = []
    for fname in ["param_count_log.npy", "synflow_log.npy",
                  "naswot_log.npy", "zenscore_log.npy"]:
        path = TRANS_DIR / fname
        if path.exists():
            arr = np.load(path).astype(np.float64)
            arr = np.where(np.isfinite(arr), arr, np.nanmedian(arr))
            cols.append(arr)
    X_raw = np.column_stack(cols)
    scaler = StandardScaler()
    X_std  = scaler.fit_transform(X_raw)
    pca    = PCA(whiten=True, random_state=SEED)
    Z      = pca.fit_transform(X_std)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, 0.99)) + 1
    n_keep = min(n_keep, Z.shape[1])
    return Z[:, :n_keep]


def build_full_pipeline():
    """Variant 4: whitened debiased features from Step 6."""
    return np.load(PCA_DIR / "whitened_features.npy").astype(np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    gt = np.load(AUDIT_DIR / "gt_accuracies.npy").astype(np.float64)
    print(f"GT loaded: n={len(gt)}\n", flush=True)

    all_results = []

    # Variant 1: size only
    print("Variant 1: size_only ...", flush=True)
    X1 = build_size_only(gt)
    all_results.append(train_and_eval(X1, gt, "size_only"))

    # Variant 2: best raw proxy
    print("\nVariant 2: best_raw ...", flush=True)
    X2 = build_best_raw(gt)
    all_results.append(train_and_eval(X2, gt, "best_raw"))

    # Variant 3: PCA on raw proxies
    print("\nVariant 3: pca_raw ...", flush=True)
    X3 = build_pca_raw(gt)
    all_results.append(train_and_eval(X3, gt, "pca_raw"))

    # Variant 4: full pipeline
    print("\nVariant 4: full_pipeline ...", flush=True)
    X4 = build_full_pipeline()
    all_results.append(train_and_eval(X4, gt, "full_pipeline"))

    # Save
    with open(OUT_DIR / "mlp_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Ablation table (compact)
    fields = ["variant", "n_features",
              "spearman_rho_global", "spearman_rho_competitive",
              "kendall_tau_global",
              "top5pct_precision", "top10pct_precision", "top1pct_precision"]
    table = [{k: r[k] for k in fields} for r in all_results]
    with open(OUT_DIR / "ablation_table.json", "w") as f:
        json.dump(table, f, indent=2)

    print(f"\nSaved mlp_results.json and ablation_table.json to {OUT_DIR}", flush=True)

    # Print ablation table summary
    print("\n=== Ablation Table ===", flush=True)
    header = f"{'Variant':<20} {'feats':>5} {'rho_g':>7} {'rho_c':>7} {'top5%':>6} {'top10%':>7}"
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for r in all_results:
        print(f"  {r['variant']:<18} {r['n_features']:>5} "
              f"{r['spearman_rho_global']:>7.4f} "
              f"{r['spearman_rho_competitive']:>7.4f} "
              f"{r['top5pct_precision']:>6.3f} "
              f"{r['top10pct_precision']:>7.3f}", flush=True)


if __name__ == "__main__":
    main()
