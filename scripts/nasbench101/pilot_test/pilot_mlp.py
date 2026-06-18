"""
Pilot: Step 7 - Surrogate MLP on pilot subset (N=20,000).

Architecture: k -> 64 -> 32 -> 1
Loss:         Pairwise ranking loss
Split:        80/10/10, seed=42
Ablation:     4 variants (size_only, best_raw, pca_raw, full_pipeline)

Input:  results/nasbench101/pilot_test/pca_whitening/whitened_features.npy
        results/nasbench101/pilot_test/transformed_proxy/
        results/nasbench101/audit/gt_accuracies.npy  (first 20k)
Output: results/nasbench101/pilot_test/surrogate_mlp/ablation_table.json
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

N_PILOT     = 20_000
SEED        = 42
LR          = 1e-3
PATIENCE    = 20
MAX_EPOCHS  = 300
BATCH_SIZE  = 1024
COMP_THRESH = 50.0

ROOT_DIR   = Path("F:/Thesis/Experimentation")
AUDIT_DIR  = ROOT_DIR / "results/nasbench101/audit"
PILOT_DIR  = ROOT_DIR / "results/nasbench101/pilot_test"
TRANS_DIR  = PILOT_DIR / "transformed_proxy"
VALID_DIR  = PILOT_DIR / "proxy_validation"
PCA_DIR    = PILOT_DIR / "pca_whitening"
OUT_DIR    = PILOT_DIR / "surrogate_mlp"
OUT_DIR.mkdir(parents=True, exist_ok=True)


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def ranking_loss(pred, tgt, n=2048):
    i = torch.randint(0, len(pred), (n,))
    j = torch.randint(0, len(pred), (n,))
    mask = tgt[i] > tgt[j]
    if not mask.any():
        return torch.tensor(0.0, requires_grad=True)
    return torch.relu(-(pred[i[mask]] - pred[j[mask]])).mean()


def train_eval(X_all, gt_all, name):
    torch.manual_seed(SEED); np.random.seed(SEED)
    n   = len(gt_all)
    idx = np.random.permutation(n)
    ntr = int(0.8*n); nval = int(0.1*n)
    itr, ival, ite = idx[:ntr], idx[ntr:ntr+nval], idx[ntr+nval:]

    Xtr  = torch.from_numpy(X_all[itr].astype(np.float32))
    ytr  = torch.from_numpy(gt_all[itr].astype(np.float32))
    Xval = torch.from_numpy(X_all[ival].astype(np.float32))
    yval = torch.from_numpy(gt_all[ival].astype(np.float32))
    Xte  = torch.from_numpy(X_all[ite].astype(np.float32))
    yte  = gt_all[ite]

    model = MLP(X_all.shape[1])
    opt   = optim.Adam(model.parameters(), lr=LR)
    best_loss, best_state, patience_count = float("inf"), None, 0

    for epoch in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for s in range(0, len(Xtr), BATCH_SIZE):
            b = perm[s:s+BATCH_SIZE]
            loss = ranking_loss(model(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = ranking_loss(model(Xval), yval).item()
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
        if patience_count >= PATIENCE:
            break

    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        pred = model(Xte).numpy()

    comp = yte > COMP_THRESH
    rho_g, _ = spearmanr(pred, yte)
    tau_g, _ = kendalltau(pred, yte)
    rho_c = float(spearmanr(pred[comp], yte[comp])[0]) if comp.sum() >= 10 else float("nan")

    def topk(p, g, f):
        k = max(1, int(f*len(g)))
        return len(set(np.argsort(p)[-k:]) & set(np.argsort(g)[-k:])) / k

    result = {
        "variant":                  name,
        "n_features":               int(X_all.shape[1]),
        "spearman_rho_global":      float(rho_g),
        "spearman_rho_competitive": float(rho_c),
        "kendall_tau_global":       float(tau_g),
        "top5pct_precision":        topk(pred, yte, 0.05),
        "top10pct_precision":       topk(pred, yte, 0.10),
        "top1pct_precision":        topk(pred, yte, 0.01),
    }
    print(f"  [{name}] rho_g={rho_g:.4f}  rho_c={rho_c:.4f}  "
          f"top5%={result['top5pct_precision']:.3f}  "
          f"top10%={result['top10pct_precision']:.3f}", flush=True)
    return result


def load_and_scale(path, reshape1d=False):
    arr = np.load(path).astype(np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nanmedian(arr))
    if reshape1d: arr = arr.reshape(-1, 1)
    return StandardScaler().fit_transform(arr)


def main():
    gt = np.load(AUDIT_DIR / "gt_accuracies.npy")[:N_PILOT].astype(np.float64)
    print(f"GT loaded: n={len(gt)}\n", flush=True)

    results = []

    # Variant 1: size only
    print("Variant 1: size_only", flush=True)
    X1 = load_and_scale(TRANS_DIR / "param_count_log.npy", reshape1d=True)
    results.append(train_eval(X1, gt, "size_only"))

    # Variant 2: best raw proxy (highest global rho)
    print("\nVariant 2: best_raw", flush=True)
    with open(VALID_DIR / "correlation_results.json") as f:
        corr = json.load(f)
    best = max(["synflow", "naswot", "zenscore"],
               key=lambda p: abs(corr.get(p, {}).get("global", {}).get("spearman_rho", 0)))
    print(f"  Best raw proxy: {best}", flush=True)
    path2 = TRANS_DIR / f"{best}_log.npy"
    X2 = load_and_scale(path2 if path2.exists() else TRANS_DIR / "param_count_log.npy",
                        reshape1d=True)
    results.append(train_eval(X2, gt, f"best_raw_{best}"))

    # Variant 3: PCA on all raw proxies (no debiasing)
    print("\nVariant 3: pca_raw", flush=True)
    raw_cols = []
    for fname in ["param_count_log.npy", "synflow_log.npy", "naswot_log.npy", "zenscore_log.npy"]:
        p = TRANS_DIR / fname
        if p.exists():
            a = np.load(p).astype(np.float64)
            raw_cols.append(np.where(np.isfinite(a), a, np.nanmedian(a)))
    Xraw = np.column_stack(raw_cols)
    Xstd = StandardScaler().fit_transform(Xraw)
    pca  = PCA(whiten=True, random_state=SEED)
    Z    = pca.fit_transform(Xstd)
    cumv = np.cumsum(pca.explained_variance_ratio_)
    nk   = min(int(np.searchsorted(cumv, 0.99)) + 1, Z.shape[1])
    X3   = Z[:, :nk]
    results.append(train_eval(X3, gt, "pca_raw"))

    # Variant 4: full pipeline (debiased + PCA whitened)
    print("\nVariant 4: full_pipeline", flush=True)
    wf_path = PCA_DIR / "whitened_features.npy"
    if wf_path.exists():
        X4 = np.load(wf_path).astype(np.float64)
        results.append(train_eval(X4, gt, "full_pipeline"))
    else:
        print("  whitened_features.npy not found -- run pilot_pipeline.py first", flush=True)

    with open(OUT_DIR / "mlp_results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(OUT_DIR / "ablation_table.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Ablation Table (Pilot N={N_PILOT}) ===", flush=True)
    hdr = f"{'Variant':<22} {'feats':>5} {'rho_g':>7} {'rho_c':>7} {'top5%':>6} {'top10%':>7}"
    print(hdr, flush=True)
    print("-" * len(hdr), flush=True)
    for r in results:
        print(f"  {r['variant']:<20} {r['n_features']:>5} "
              f"{r['spearman_rho_global']:>7.4f} "
              f"{r['spearman_rho_competitive']:>7.4f} "
              f"{r['top5pct_precision']:>6.3f} "
              f"{r['top10pct_precision']:>7.3f}", flush=True)

    print(f"\nSaved to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
