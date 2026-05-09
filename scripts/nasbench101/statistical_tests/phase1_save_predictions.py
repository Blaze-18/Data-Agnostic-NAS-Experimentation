"""
Statistical Validation — Phase 1: Save Test-Set Predictions
============================================================
Prerequisite for Tests 1 (bootstrap CI) and 3 (competitive subset analysis).

Replicates the exact feature matrices and seed-42 train/val/test split from
train_mlp_101.py, retrains pca_raw and full_pipeline, then saves:

    test_pred_pca_raw.npy       (n_test,)  float32 — model scores on test set
    test_pred_full_pipeline.npy (n_test,)  float32
    test_gt.npy                 (n_test,)  float64 — ground-truth accuracies
    test_indices.npy            (n_test,)  int64   — indices into the full 423k array

Output: results/nasbench101/statistical_validation/predictions/
"""

import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths  (must exactly match train_mlp_101.py)
# ---------------------------------------------------------------------------
ROOT_DIR  = Path(__file__).resolve().parents[3]
AUDIT_DIR = ROOT_DIR / "results/nasbench101/audit"
TRANS_DIR = ROOT_DIR / "results/nasbench101/transformed_proxy"
VALID_DIR = ROOT_DIR / "results/nasbench101/proxy_validation"
PCA_DIR   = ROOT_DIR / "results/nasbench101/pca_whitening"
OUT_DIR   = ROOT_DIR / "results/nasbench101/statistical_validation/predictions"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters  (must exactly match train_mlp_101.py)
# ---------------------------------------------------------------------------
SEED       = 42
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
LR         = 1e-3
PATIENCE   = 20
MAX_EPOCHS = 500
BATCH_SIZE = 2048


# ---------------------------------------------------------------------------
# MLP  (identical architecture to train_mlp_101.py)
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Loss  (identical to train_mlp_101.py — RankNet log-sigmoid)
# ---------------------------------------------------------------------------
def ranknet_loss(pred, target, n_pairs=2048):
    n_pairs = min(n_pairs, len(pred))
    idx_i = torch.randint(0, len(pred), (n_pairs,), device=pred.device)
    idx_j = torch.randint(0, len(pred), (n_pairs,), device=pred.device)
    mask  = target[idx_i] > target[idx_j]
    if mask.sum() == 0:
        return pred.sum() * 0.0 + torch.log(torch.tensor(2.0, device=pred.device))
    diff = pred[idx_i[mask]] - pred[idx_j[mask]]
    return torch.nn.functional.softplus(-diff).mean()


# ---------------------------------------------------------------------------
# Feature builders  (exact copies from train_mlp_101.py)
# ---------------------------------------------------------------------------
def build_pca_raw():
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
    return np.load(PCA_DIR / "whitened_features.npy").astype(np.float64)


# ---------------------------------------------------------------------------
# Train + return test-set predictions
# ---------------------------------------------------------------------------
def train_and_predict(X_all: np.ndarray, gt_all: np.ndarray,
                      variant_name: str):
    """Train MLP with the canonical seed-42 split; return (test_pred, test_idx)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n       = len(gt_all)
    idx     = np.random.permutation(n)
    n_train = int(TRAIN_FRAC * n)
    n_val   = int(VAL_FRAC   * n)

    idx_train = idx[:n_train]
    idx_val   = idx[n_train:n_train + n_val]
    idx_test  = idx[n_train + n_val:]

    X_tr  = torch.from_numpy(X_all[idx_train].astype(np.float32)).to(device)
    y_tr  = torch.from_numpy(gt_all[idx_train].astype(np.float32)).to(device)
    X_val = torch.from_numpy(X_all[idx_val].astype(np.float32)).to(device)
    y_val = torch.from_numpy(gt_all[idx_val].astype(np.float32)).to(device)
    X_te  = torch.from_numpy(X_all[idx_test].astype(np.float32)).to(device)

    model     = MLP(in_dim=X_all.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm   = torch.randperm(len(X_tr), device=device)
        for start in range(0, len(X_tr), BATCH_SIZE):
            batch_idx = perm[start:start + BATCH_SIZE]
            xb, yb    = X_tr[batch_idx], y_tr[batch_idx]
            loss      = ranknet_loss(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = ranknet_loss(model(X_val), y_val).item()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print(f"  [{variant_name}] Early stop epoch={epoch+1}  "
                  f"val_loss={best_val_loss:.4f}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_pred = model(X_te).cpu().numpy()

    return test_pred, idx_test


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    gt = np.load(AUDIT_DIR / "gt_accuracies.npy").astype(np.float64)
    print(f"GT loaded: n={len(gt)}\n", flush=True)

    # pca_raw
    print("Training pca_raw ...", flush=True)
    X_pca = build_pca_raw()
    pred_pca, idx_test = train_and_predict(X_pca, gt, "pca_raw")

    # full_pipeline  — must use the SAME idx_test (same seed guarantees this,
    # but we double-check by reusing the indices from pca_raw)
    print("\nTraining full_pipeline ...", flush=True)
    X_fp = build_full_pipeline()
    pred_fp, idx_test_fp = train_and_predict(X_fp, gt, "full_pipeline")

    assert np.array_equal(idx_test, idx_test_fp), \
        "Test indices differ between variants — seed not deterministic!"

    # Save
    np.save(OUT_DIR / "test_pred_pca_raw.npy",        pred_pca.astype(np.float32))
    np.save(OUT_DIR / "test_pred_full_pipeline.npy",   pred_fp.astype(np.float32))
    np.save(OUT_DIR / "test_gt.npy",                   gt[idx_test].astype(np.float64))
    np.save(OUT_DIR / "test_indices.npy",              idx_test.astype(np.int64))

    print(f"\nSaved predictions ({len(idx_test)} test architectures) to {OUT_DIR}")
    print("  test_pred_pca_raw.npy")
    print("  test_pred_full_pipeline.npy")
    print("  test_gt.npy")
    print("  test_indices.npy")
    print("\nPhase 1 complete. Run phase2_statistical_tests.py next.")


if __name__ == "__main__":
    main()
