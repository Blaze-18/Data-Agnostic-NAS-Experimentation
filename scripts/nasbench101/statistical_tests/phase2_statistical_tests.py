"""
Statistical Validation — Phase 2: Bootstrap CI + Competitive Subset Analysis
=============================================================================
Requires predictions saved by phase1_save_predictions.py.

Test 1 — Bootstrap Confidence Interval on the debiasing contribution:
    1000 bootstrap resamples of the test set.
    For each resample: Spearman ρ(pca_raw) and ρ(full_pipeline).
    Reports 95% CI of the difference Δρ = ρ_full - ρ_pca.
    Saves: bootstrap_results.json, bootstrap_delta_histogram.png

Test 3 — Strict Competitive Subset Analysis:
    Computes Spearman ρ for all 4 ablation variants at three GT thresholds:
        GT > 50%  (existing competitive subset)
        GT > 85%  (genuinely competitive range)
        GT > 90%  (hard high-performance tail)
    Also loads size_only and best_raw predictions by re-inferring from
    re-trained models OR loads from mlp_results.json for the global ρ values.
    Since size_only/best_raw predictions are not saved, this script
    re-trains them briefly OR works from saved mlp_results.json rho values.
    For the subset analysis only pca_raw and full_pipeline predictions are
    strictly needed (the core comparison); size_only and best_raw are
    included for completeness using their global ρ as a reference.
    Saves: competitive_subset_analysis.json, competitive_subset_plot.png

Input:  results/nasbench101/statistical_validation/predictions/
Output: results/nasbench101/statistical_validation/
"""

import numpy as np
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR  = Path(__file__).resolve().parents[3]
PRED_DIR  = ROOT_DIR / "results/nasbench101/statistical_validation/predictions"
MLP_DIR   = ROOT_DIR / "results/nasbench101/surrogate_mlp"
OUT_DIR   = ROOT_DIR / "results/nasbench101/statistical_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Rebuild size_only and best_raw predictions for subset analysis
AUDIT_DIR = ROOT_DIR / "results/nasbench101/audit"
TRANS_DIR = ROOT_DIR / "results/nasbench101/transformed_proxy"
VALID_DIR = ROOT_DIR / "results/nasbench101/proxy_validation"
PCA_DIR   = ROOT_DIR / "results/nasbench101/pca_whitening"

SEED           = 42
TRAIN_FRAC     = 0.80
VAL_FRAC       = 0.10
N_BOOTSTRAP    = 1000
THRESHOLDS     = [50.0, 85.0, 90.0]


# ---------------------------------------------------------------------------
# Helper: load saved predictions
# ---------------------------------------------------------------------------
def load_predictions():
    pred_pca = np.load(PRED_DIR / "test_pred_pca_raw.npy").astype(np.float64)
    pred_fp  = np.load(PRED_DIR / "test_pred_full_pipeline.npy").astype(np.float64)
    gt_te    = np.load(PRED_DIR / "test_gt.npy").astype(np.float64)
    idx_te   = np.load(PRED_DIR / "test_indices.npy")
    return pred_pca, pred_fp, gt_te, idx_te


# ---------------------------------------------------------------------------
# Helper: rebuild size_only and best_raw predictions using the same test split
# (re-trains the two lightweight 1-feature models — takes ~1 min on GPU)
# ---------------------------------------------------------------------------
def _build_size_only_features():
    from sklearn.preprocessing import StandardScaler
    arr = np.load(TRANS_DIR / "param_count_log.npy").astype(np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nanmedian(arr))
    return StandardScaler().fit_transform(arr.reshape(-1, 1))


def _build_best_raw_features():
    from sklearn.preprocessing import StandardScaler
    with open(VALID_DIR / "correlation_results.json") as f:
        corr = json.load(f)
    candidates = ["synflow", "naswot", "zenscore"]
    file_map   = {"synflow": "synflow_log.npy",
                  "naswot":  "naswot_log.npy",
                  "zenscore":"zenscore_log.npy"}
    best = max(candidates,
               key=lambda p: abs(corr.get(p, {}).get("global", {})
                                 .get("spearman_rho", 0.0)))
    arr = np.load(TRANS_DIR / file_map[best]).astype(np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nanmedian(arr))
    return StandardScaler().fit_transform(arr.reshape(-1, 1)), best


def rebuild_baseline_predictions(gt_all, idx_test):
    """Retrain size_only and best_raw, return their test-set predictions."""
    import torch
    import torch.nn as nn
    import torch.optim as optim

    class _MLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 64), nn.ReLU(),
                nn.Linear(64, 32),     nn.ReLU(),
                nn.Linear(32, 1),
            )
        def forward(self, x): return self.net(x).squeeze(-1)

    def _ranknet(pred, target, n_pairs=2048):
        n_pairs = min(n_pairs, len(pred))
        ii = torch.randint(0, len(pred), (n_pairs,), device=pred.device)
        jj = torch.randint(0, len(pred), (n_pairs,), device=pred.device)
        mask = target[ii] > target[jj]
        if mask.sum() == 0:
            return pred.sum() * 0.0 + torch.log(torch.tensor(2.0, device=pred.device))
        return torch.nn.functional.softplus(-(pred[ii[mask]] - pred[jj[mask]])).mean()

    def _train(X_all, gt_all, idx_test, name):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(SEED); np.random.seed(SEED)
        n = len(gt_all)
        idx = np.random.permutation(n)
        n_tr = int(TRAIN_FRAC * n); n_val = int(VAL_FRAC * n)
        X_tr  = torch.from_numpy(X_all[idx[:n_tr]].astype(np.float32)).to(device)
        y_tr  = torch.from_numpy(gt_all[idx[:n_tr]].astype(np.float32)).to(device)
        X_val = torch.from_numpy(X_all[idx[n_tr:n_tr+n_val]].astype(np.float32)).to(device)
        y_val = torch.from_numpy(gt_all[idx[n_tr:n_tr+n_val]].astype(np.float32)).to(device)
        X_te  = torch.from_numpy(X_all[idx_test].astype(np.float32)).to(device)
        model = _MLP(X_all.shape[1]).to(device)
        opt   = optim.Adam(model.parameters(), lr=1e-3)
        best_loss, best_state, patience = float("inf"), None, 0
        for ep in range(500):
            model.train()
            perm = torch.randperm(len(X_tr), device=device)
            for s in range(0, len(X_tr), 2048):
                bi = perm[s:s+2048]; xb, yb = X_tr[bi], y_tr[bi]
                loss = _ranknet(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                vl = _ranknet(model(X_val), y_val).item()
            if vl < best_loss:
                best_loss = vl
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
            if patience >= 20:
                print(f"  [{name}] Early stop epoch={ep+1}  val_loss={best_loss:.4f}",
                      flush=True)
                break
        model.load_state_dict(best_state); model.eval()
        with torch.no_grad():
            return model(X_te).cpu().numpy()

    print("  Rebuilding size_only predictions ...", flush=True)
    X_so = _build_size_only_features()
    pred_so = _train(X_so, gt_all, idx_test, "size_only")

    print("  Rebuilding best_raw predictions ...", flush=True)
    X_br, best_name = _build_best_raw_features()
    print(f"    best_raw proxy = {best_name}", flush=True)
    pred_br = _train(X_br, gt_all, idx_test, "best_raw")

    return pred_so, pred_br


# ---------------------------------------------------------------------------
# Test 1: Bootstrap CI
# ---------------------------------------------------------------------------
def run_bootstrap(pred_pca, pred_fp, gt_te):
    print(f"\n{'='*60}", flush=True)
    print(f"Test 1: Bootstrap CI  (n_bootstrap={N_BOOTSTRAP})", flush=True)
    print(f"{'='*60}", flush=True)

    rng = np.random.default_rng(SEED)
    n   = len(gt_te)
    deltas, rhos_pca, rhos_fp = [], [], []

    for i in range(N_BOOTSTRAP):
        idx = rng.choice(n, size=n, replace=True)
        rho_p, _ = spearmanr(pred_pca[idx], gt_te[idx])
        rho_f, _ = spearmanr(pred_fp[idx],  gt_te[idx])
        rhos_pca.append(rho_p)
        rhos_fp.append(rho_f)
        deltas.append(rho_f - rho_p)
        if (i + 1) % 200 == 0:
            print(f"  ... {i+1}/{N_BOOTSTRAP} resamples done", flush=True)

    deltas   = np.array(deltas)
    rhos_pca = np.array(rhos_pca)
    rhos_fp  = np.array(rhos_fp)

    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    mean_delta   = float(np.mean(deltas))
    p_positive   = float(np.mean(deltas > 0))
    point_rho_p, _ = spearmanr(pred_pca, gt_te)
    point_rho_f, _ = spearmanr(pred_fp,  gt_te)
    point_delta  = float(point_rho_f - point_rho_p)

    result = {
        "n_test":              int(n),
        "n_bootstrap":         N_BOOTSTRAP,
        "point_rho_pca_raw":   float(point_rho_p),
        "point_rho_full_pipeline": float(point_rho_f),
        "point_delta":         point_delta,
        "bootstrap_mean_delta": mean_delta,
        "ci_95_lower":         float(ci_lo),
        "ci_95_upper":         float(ci_hi),
        "ci_excludes_zero":    bool(ci_lo > 0),
        "p_delta_positive":    p_positive,
    }

    print(f"\n  Point estimates:")
    print(f"    pca_raw  ρ = {point_rho_p:.4f}")
    print(f"    full_pipeline ρ = {point_rho_f:.4f}")
    print(f"    Δρ (point) = {point_delta:+.4f}")
    print(f"\n  Bootstrap results:")
    print(f"    Mean Δρ  = {mean_delta:+.4f}")
    print(f"    95% CI   = [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    print(f"    CI excludes zero: {result['ci_excludes_zero']}")
    print(f"    P(Δρ > 0): {p_positive:.4f}")

    # Save JSON
    with open(OUT_DIR / "bootstrap_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved bootstrap_results.json", flush=True)

    # Plot histogram
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(deltas, bins=50, color="#3498db", alpha=0.8, edgecolor="white", lw=0.4)
    ax.axvline(mean_delta, color="#e74c3c", lw=2.0, label=f"Mean Δρ = {mean_delta:+.4f}")
    ax.axvline(ci_lo, color="#e74c3c", lw=1.5, ls="--",
               label=f"95% CI [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    ax.axvline(ci_hi, color="#e74c3c", lw=1.5, ls="--")
    ax.axvline(0, color="black", lw=1.0, ls=":", label="Δρ = 0")
    ax.set_xlabel("Δρ  (full_pipeline − pca_raw)", fontsize=11)
    ax.set_ylabel("Bootstrap frequency", fontsize=11)
    ax.set_title("Bootstrap Distribution of Debiasing Contribution\n"
                 "NAS-Bench-101 — Test Set (n≈42k, 1000 resamples)", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bootstrap_delta_histogram.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bootstrap_delta_histogram.png", flush=True)

    return result


# ---------------------------------------------------------------------------
# Test 3: Strict Competitive Subset Analysis
# ---------------------------------------------------------------------------
def run_competitive_subset(pred_so, pred_br, pred_pca, pred_fp, gt_te):
    print(f"\n{'='*60}", flush=True)
    print(f"Test 3: Strict Competitive Subset Analysis", flush=True)
    print(f"{'='*60}", flush=True)

    variants = {
        "size_only":     pred_so,
        "best_raw":      pred_br,
        "pca_raw":       pred_pca,
        "full_pipeline": pred_fp,
    }

    records = []
    for thresh in THRESHOLDS:
        mask = gt_te > thresh
        n_sub = int(mask.sum())
        row = {"threshold": thresh, "n_subset": n_sub}
        print(f"\n  GT > {thresh}%  →  n = {n_sub}", flush=True)
        for vname, pred in variants.items():
            if n_sub >= 10:
                rho, _ = spearmanr(pred[mask], gt_te[mask])
            else:
                rho = float("nan")
            row[f"rho_{vname}"] = float(rho)
            print(f"    {vname:<18}: ρ = {rho:.4f}", flush=True)
        row["gap_full_vs_pca"] = row["rho_full_pipeline"] - row["rho_pca_raw"]
        print(f"    {'gap (full−pca)':<18}: {row['gap_full_vs_pca']:+.4f}", flush=True)
        records.append(row)

    result = {"thresholds_analysis": records}
    with open(OUT_DIR / "competitive_subset_analysis.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  Saved competitive_subset_analysis.json", flush=True)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: ρ per variant per threshold
    ax = axes[0]
    x  = np.arange(len(THRESHOLDS))
    colors = {"size_only": "#95a5a6", "best_raw": "#e67e22",
              "pca_raw": "#3498db",    "full_pipeline": "#2ecc71"}
    w  = 0.20
    for i, (vname, _) in enumerate(variants.items()):
        rhos = [r[f"rho_{vname}"] for r in records]
        ax.bar(x + (i - 1.5) * w, rhos, w,
               label=vname.replace("_", " "), color=colors[vname], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"GT > {t}%" for t in THRESHOLDS], fontsize=9)
    ax.set_ylabel("Spearman ρ", fontsize=10)
    ax.set_title("Spearman ρ by Competitive Subset Threshold", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.0)

    # Right: gap (full_pipeline − pca_raw) per threshold
    ax2 = axes[1]
    gaps  = [r["gap_full_vs_pca"] for r in records]
    sizes = [r["n_subset"] for r in records]
    bars  = ax2.bar(x, gaps, 0.4, color="#9b59b6", alpha=0.85)
    ax2.axhline(0, color="black", lw=0.8, ls=":")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"GT > {t}%\n(n={s:,})" for t, s in zip(THRESHOLDS, sizes)],
                        fontsize=8)
    ax2.set_ylabel("Δρ  (full_pipeline − pca_raw)", fontsize=10)
    ax2.set_title("Debiasing Gap Across Difficulty Thresholds", fontsize=10, fontweight="bold")
    for bar, g in zip(bars, gaps):
        ax2.text(bar.get_x() + bar.get_width() / 2, g + 0.001,
                 f"{g:+.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("NAS-Bench-101 — Competitive Subset Analysis", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "competitive_subset_plot.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved competitive_subset_plot.png", flush=True)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    # Load predictions
    print("Loading saved predictions ...", flush=True)
    pred_pca, pred_fp, gt_te, idx_te = load_predictions()
    print(f"  Test set size: {len(gt_te)}", flush=True)

    # Rebuild size_only and best_raw for subset analysis
    gt_all = np.load(AUDIT_DIR / "gt_accuracies.npy").astype(np.float64)
    print("\nRebuilding size_only + best_raw for subset analysis ...", flush=True)
    pred_so, pred_br = rebuild_baseline_predictions(gt_all, idx_te)

    # Test 1: Bootstrap CI
    boot_result = run_bootstrap(pred_pca, pred_fp, gt_te)

    # Test 3: Competitive subset analysis
    subset_result = run_competitive_subset(pred_so, pred_br, pred_pca, pred_fp, gt_te)

    # Combined summary
    summary = {
        "bootstrap": boot_result,
        "competitive_subset": subset_result,
    }
    with open(OUT_DIR / "phase2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Phase 2 complete. All outputs in:")
    print(f"  {OUT_DIR}")
    print(f"  bootstrap_results.json")
    print(f"  bootstrap_delta_histogram.png")
    print(f"  competitive_subset_analysis.json")
    print(f"  competitive_subset_plot.png")
    print(f"  phase2_summary.json")


if __name__ == "__main__":
    main()
