"""
Statistical Validation — Phase 3: Multi-Seed Stability Check
=============================================================
Runs pca_raw and full_pipeline MLP training across 5 seeds
(0, 1, 2, 3, 42) using identical hyperparameters. Seed 42 results
are loaded from the existing mlp_results.json rather than re-trained.

Reports:
    Per-seed ρ_global for pca_raw and full_pipeline
    Gap = full_pipeline ρ − pca_raw ρ per seed
    Mean ± std across all 5 seeds

Output: results/nasbench101/statistical_validation/
    multi_seed_results.json
    multi_seed_gap_plot.png

Usage:
    python phase3_multi_seed.py
    python phase3_multi_seed.py --seeds 0 1 2 3   # skip 42 (already done)
"""

import argparse
import numpy as np
import json
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR  = Path(__file__).resolve().parents[3]
AUDIT_DIR = ROOT_DIR / "results/nasbench101/audit"
TRANS_DIR = ROOT_DIR / "results/nasbench101/transformed_proxy"
PCA_DIR   = ROOT_DIR / "results/nasbench101/pca_whitening"
MLP_DIR   = ROOT_DIR / "results/nasbench101/surrogate_mlp"
OUT_DIR   = ROOT_DIR / "results/nasbench101/statistical_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameters  (must match train_mlp_101.py exactly)
# ---------------------------------------------------------------------------
TRAIN_FRAC = 0.80
VAL_FRAC   = 0.10
LR         = 1e-3
PATIENCE   = 20
MAX_EPOCHS = 500
BATCH_SIZE = 2048
COMP_THRESH = 50.0

ALL_SEEDS  = [0, 1, 2, 3, 4, 5, 6, 7, 42]


# ---------------------------------------------------------------------------
# MLP
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 32),     nn.ReLU(),
            nn.Linear(32, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)


def ranknet_loss(pred, target, n_pairs=2048):
    n_pairs = min(n_pairs, len(pred))
    ii = torch.randint(0, len(pred), (n_pairs,), device=pred.device)
    jj = torch.randint(0, len(pred), (n_pairs,), device=pred.device)
    mask = target[ii] > target[jj]
    if mask.sum() == 0:
        return pred.sum() * 0.0 + torch.log(torch.tensor(2.0, device=pred.device))
    return torch.nn.functional.softplus(-(pred[ii[mask]] - pred[jj[mask]])).mean()


# ---------------------------------------------------------------------------
# Feature builders  (exact copies — PCA must be re-fit per seed because
# train/val split changes; the raw proxies are fixed)
#
# *** Intentional asymmetry ***
# pca_raw:       PCA is re-fit with random_state=seed on each run, so the
#                embedding itself varies slightly across seeds (different
#                random tie-breaking in the SVD).
# full_pipeline: always loads the fixed whitened_features.npy produced by
#                Step 6 (PCA fitted once with random_state=42). Seed
#                variation for this variant therefore comes only from the
#                train/val/test split permutation and weight initialisation,
#                NOT from PCA fitting.
# This asymmetry is defensible: the full_pipeline embedding is a fixed,
# pre-computed artefact; we test how stable the downstream MLP is to
# different splits and initialisations given that fixed input.
# ---------------------------------------------------------------------------
def build_pca_raw(seed):
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
    pca    = PCA(whiten=True, random_state=seed)
    Z      = pca.fit_transform(X_std)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    n_keep = int(np.searchsorted(cumvar, 0.99)) + 1
    return Z[:, :min(n_keep, Z.shape[1])]


def build_full_pipeline():
    """Full pipeline features are fixed (debiasing + PCA done in Step 6)."""
    return np.load(PCA_DIR / "whitened_features.npy").astype(np.float64)


# ---------------------------------------------------------------------------
# Train one variant with a given seed
# ---------------------------------------------------------------------------
def train_one(X_all, gt_all, variant_name, seed):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    n     = len(gt_all)
    idx   = np.random.permutation(n)
    n_tr  = int(TRAIN_FRAC * n)
    n_val = int(VAL_FRAC   * n)

    idx_tr  = idx[:n_tr]
    idx_val = idx[n_tr:n_tr + n_val]
    idx_te  = idx[n_tr + n_val:]

    X_tr  = torch.from_numpy(X_all[idx_tr].astype(np.float32)).to(device)
    y_tr  = torch.from_numpy(gt_all[idx_tr].astype(np.float32)).to(device)
    X_val = torch.from_numpy(X_all[idx_val].astype(np.float32)).to(device)
    y_val = torch.from_numpy(gt_all[idx_val].astype(np.float32)).to(device)
    X_te  = torch.from_numpy(X_all[idx_te].astype(np.float32)).to(device)
    y_te  = gt_all[idx_te]

    model = MLP(X_all.shape[1]).to(device)
    opt   = optim.Adam(model.parameters(), lr=LR)

    best_loss, best_state, patience = float("inf"), None, 0

    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(X_tr), device=device)
        for s in range(0, len(X_tr), BATCH_SIZE):
            bi = perm[s:s + BATCH_SIZE]
            loss = ranknet_loss(model(X_tr[bi]), y_tr[bi])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            vl = ranknet_loss(model(X_val), y_val).item()

        if vl < best_loss:
            best_loss  = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience   = 0
        else:
            patience += 1

        if patience >= PATIENCE:
            print(f"    [seed={seed} | {variant_name}] "
                  f"Early stop epoch={ep+1}  val_loss={best_loss:.4f}", flush=True)
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_te = model(X_te).cpu().numpy()

    comp_mask  = y_te > COMP_THRESH
    rho_g, _   = spearmanr(pred_te, y_te)
    rho_c      = float("nan")
    if comp_mask.sum() >= 10:
        rho_c, _ = spearmanr(pred_te[comp_mask], y_te[comp_mask])

    print(f"    [seed={seed} | {variant_name}] "
          f"ρ_global={rho_g:.4f}  ρ_comp={rho_c:.4f}", flush=True)

    return float(rho_g), float(rho_c)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to run (default: all 5). Seed 42 is always "
                             "loaded from existing mlp_results.json.")
    args = parser.parse_args()

    seeds_to_run = args.seeds if args.seeds is not None else ALL_SEEDS

    gt = np.load(AUDIT_DIR / "gt_accuracies.npy").astype(np.float64)
    print(f"GT loaded: n={len(gt)}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n", flush=True)

    # Load seed-42 result from existing JSON (avoid retraining)
    existing = {}
    mlp_json = MLP_DIR / "mlp_results.json"
    if mlp_json.exists():
        with open(mlp_json) as f:
            for r in json.load(f):
                existing[r["variant"]] = r
        print(f"Loaded seed-42 results from mlp_results.json", flush=True)

    per_seed = {}  # {seed: {pca_raw: {rho_g, rho_c}, full_pipeline: {...}}}

    # Inject seed-42 from existing run
    if 42 in seeds_to_run or 42 not in seeds_to_run:
        if "pca_raw" in existing and "full_pipeline" in existing:
            per_seed[42] = {
                "pca_raw": {
                    "rho_global":      existing["pca_raw"]["spearman_rho_global"],
                    "rho_competitive": existing["pca_raw"]["spearman_rho_competitive"],
                },
                "full_pipeline": {
                    "rho_global":      existing["full_pipeline"]["spearman_rho_global"],
                    "rho_competitive": existing["full_pipeline"]["spearman_rho_competitive"],
                },
            }
            print(f"  Seed 42: loaded from existing run (no retraining)\n", flush=True)

    # Run remaining seeds
    new_seeds = [s for s in seeds_to_run if s != 42]
    for seed in new_seeds:
        print(f"\n{'─'*50}", flush=True)
        print(f"Seed {seed}", flush=True)
        print(f"{'─'*50}", flush=True)

        print(f"  Building pca_raw features ...", flush=True)
        X_pca = build_pca_raw(seed)

        print(f"  Building full_pipeline features ...", flush=True)
        X_fp  = build_full_pipeline()

        rho_g_pca, rho_c_pca = train_one(X_pca, gt, "pca_raw",       seed)
        rho_g_fp,  rho_c_fp  = train_one(X_fp,  gt, "full_pipeline",  seed)

        per_seed[seed] = {
            "pca_raw":       {"rho_global": rho_g_pca, "rho_competitive": rho_c_pca},
            "full_pipeline": {"rho_global": rho_g_fp,  "rho_competitive": rho_c_fp},
        }

    # Merge with any previously saved results
    existing_json = OUT_DIR / "multi_seed_results.json"
    if existing_json.exists():
        with open(existing_json) as f:
            prev = json.load(f)
        for s_str, v in prev.get("per_seed", {}).items():
            s = int(s_str)
            if s not in per_seed:
                per_seed[s] = v
        print(f"\nMerged with previous run — total seeds: {sorted(per_seed.keys())}", flush=True)

    # Compute summary
    seeds_done = sorted(per_seed.keys())
    gaps   = [per_seed[s]["full_pipeline"]["rho_global"] -
               per_seed[s]["pca_raw"]["rho_global"] for s in seeds_done]
    rhos_pca = [per_seed[s]["pca_raw"]["rho_global"]       for s in seeds_done]
    rhos_fp  = [per_seed[s]["full_pipeline"]["rho_global"] for s in seeds_done]

    summary = {
        "seeds_run":       seeds_done,
        "per_seed":        {str(s): per_seed[s] for s in seeds_done},
        "mean_rho_pca_raw":       float(np.mean(rhos_pca)),
        "std_rho_pca_raw":        float(np.std(rhos_pca)),
        "mean_rho_full_pipeline": float(np.mean(rhos_fp)),
        "std_rho_full_pipeline":  float(np.std(rhos_fp)),
        "mean_gap":               float(np.mean(gaps)),
        "std_gap":                float(np.std(gaps)),
        "min_gap":                float(np.min(gaps)),
        "max_gap":                float(np.max(gaps)),
        "gap_always_positive":    bool(np.all(np.array(gaps) > 0)),
        "n_seeds_gap_positive":   int(np.sum(np.array(gaps) > 0)),
    }

    with open(OUT_DIR / "multi_seed_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print table
    print(f"\n{'='*60}")
    print(f"Multi-Seed Stability Results")
    print(f"{'='*60}")
    header = f"  {'Seed':>6} | {'pca_raw ρ':>10} | {'full_pipe ρ':>11} | {'Gap':>8}"
    print(header)
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}")
    for s, rp, rf, g in zip(seeds_done, rhos_pca, rhos_fp, gaps):
        marker = " *" if s == 42 else ""
        print(f"  {s:>6} | {rp:>10.4f} | {rf:>11.4f} | {g:>+8.4f}{marker}")
    print(f"  {'─'*6}-+-{'─'*10}-+-{'─'*11}-+-{'─'*8}")
    print(f"  {'mean':>6} | {np.mean(rhos_pca):>10.4f} | {np.mean(rhos_fp):>11.4f} | "
          f"{np.mean(gaps):>+8.4f}")
    print(f"  {'std':>6} | {np.std(rhos_pca):>10.4f} | {np.std(rhos_fp):>11.4f} | "
          f"{np.std(gaps):>8.4f}")
    print(f"\n  * = loaded from existing seed-42 run")
    print(f"\n  Gap positive in {summary['n_seeds_gap_positive']}/{len(seeds_done)} seeds")
    print(f"  Gap range: [{np.min(gaps):+.4f}, {np.max(gaps):+.4f}]")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: ρ per seed for both variants
    ax = axes[0]
    x  = np.arange(len(seeds_done))
    ax.plot(x, rhos_pca, "o--", color="#3498db", lw=1.8, ms=7, label="pca_raw")
    ax.plot(x, rhos_fp,  "s-",  color="#2ecc71", lw=1.8, ms=7, label="full_pipeline")
    ax.fill_between(x, rhos_pca, rhos_fp, alpha=0.12, color="#9b59b6",
                    label="debiasing gap")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds_done])
    ax.set_xlabel("Random seed", fontsize=10)
    ax.set_ylabel("Spearman ρ (global)", fontsize=10)
    ax.set_title("ρ Stability Across Seeds", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.set_ylim(max(0, min(rhos_pca) - 0.05), min(1.0, max(rhos_fp) + 0.05))

    # Right: gap per seed with mean ± std band
    ax2 = axes[1]
    bar_colors = ["#e74c3c" if g <= 0 else "#9b59b6" for g in gaps]
    ax2.bar(x, gaps, 0.4, color=bar_colors, alpha=0.85)
    mean_g = float(np.mean(gaps))
    std_g  = float(np.std(gaps))
    ax2.axhline(mean_g, color="#2c3e50", lw=2.0, ls="-",  label=f"Mean = {mean_g:+.4f}")
    ax2.axhline(mean_g + std_g, color="#2c3e50", lw=1.2, ls="--",
                label=f"±1 SD = {std_g:.4f}")
    ax2.axhline(mean_g - std_g, color="#2c3e50", lw=1.2, ls="--")
    ax2.axhline(0, color="black", lw=0.8, ls=":")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in seeds_done])
    ax2.set_xlabel("Random seed", fontsize=10)
    ax2.set_ylabel("Δρ  (full_pipeline − pca_raw)", fontsize=10)
    ax2.set_title("Debiasing Gap Across Seeds", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)

    for xi, g in zip(x, gaps):
        ax2.text(xi, g + (0.001 if g >= 0 else -0.003),
                 f"{g:+.3f}", ha="center", va="bottom" if g >= 0 else "top",
                 fontsize=8, fontweight="bold")

    fig.suptitle("NAS-Bench-101 — Multi-Seed Stability (pca_raw vs full_pipeline)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multi_seed_gap_plot.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Saved multi_seed_results.json")
    print(f"  Saved multi_seed_gap_plot.png")
    print(f"\nPhase 3 complete.")


if __name__ == "__main__":
    main()
