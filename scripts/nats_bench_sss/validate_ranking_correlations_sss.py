"""
Step 4 -- Ranking Correlation Validation (NATS-Bench SSS).

Computes Spearman rho, Kendall tau, and top-K precision for each proxy
against ground-truth CIFAR-10 accuracy.  Reports both raw and log-transformed
scores.

SSS-specific notes:
  - GT key: data['90']['all_results'][('cifar10', 777)]['eval_acc1es']['ori-test@89']
  - N=32,768 architectures (all evaluated -- no degenerate 'none'-op cluster)
  - Proxy files: results/nats_bench_sss/{raw_proxy_scores,transformed_proxy}/{name}.npy
  - Expected global rho values lower than NAS-Bench-201 because there is no
    easy-to-rank degenerate cluster inflating correlations.

Outputs (results/nats_bench_sss/proxy_validation/):
  correlation_results.json
  correlation_table.csv  (if pandas available)
  scatter_plots.png
  topk_precision.png
  accuracy_distribution.png
"""

import os
import json
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from scipy.stats import spearmanr, kendalltau

ARCH_DIR    = Path("data/nats_bench_sss/NATS-sss-v1_0-50262-simple")
RAW_DIR     = Path("results/nats_bench_sss/raw_proxy_scores")
TRANS_DIR   = Path("results/nats_bench_sss/transformed_proxy")
OUT_DIR     = Path("results/nats_bench_sss/proxy_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_ARCHS     = 32_768
PROXY_NAMES = ["param_count", "naswot", "zenscore", "synflow"]
DISPLAY     = {
    "param_count": "Param Count",
    "naswot":      "NASWOT",
    "zenscore":    "Zen-Score",
    "synflow":     "SynFlow",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_gt_accuracy() -> np.ndarray:
    """Load GT CIFAR-10 accuracy for all 32,768 architectures.
    Returns array indexed 0..32767."""
    gt = np.full(N_ARCHS, np.nan)
    print("  Loading GT accuracy ...")
    for idx in range(N_ARCHS):
        if idx % 5000 == 0:
            print(f"    {idx}/{N_ARCHS}", flush=True)
        try:
            with open(ARCH_DIR / f"{idx}.pickle", "rb") as f:
                data = pickle.load(f)
            r = data["90"]["all_results"][("cifar10", 777)]
            gt[idx] = float(r["eval_acc1es"]["ori-test@89"])
        except Exception:
            pass
    n_valid = int(np.sum(np.isfinite(gt)))
    print(f"  GT loaded: {n_valid}/{N_ARCHS} valid")
    return gt


def load_proxy(name: str, kind: str) -> np.ndarray:
    """Load a proxy .npy array. kind='raw' or 'transformed'."""
    base = RAW_DIR if kind == "raw" else TRANS_DIR
    return np.load(base / f"{name}.npy")


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

def compute_correlations(scores: np.ndarray, gt: np.ndarray,
                         proxy_name: str) -> dict:
    mask = np.isfinite(scores) & np.isfinite(gt)
    s, g = scores[mask], gt[mask]
    n = int(mask.sum())
    if n < 2:
        return {"proxy_name": proxy_name, "count": n,
                "spearman_rho": float("nan"), "spearman_pvalue": float("nan"),
                "kendall_tau":  float("nan"), "kendall_pvalue":  float("nan"),
                "interpretation": "N/A",
                "topk_precision_1pct":  float("nan"),
                "topk_precision_5pct":  float("nan"),
                "topk_precision_10pct": float("nan")}

    rho, rho_p  = spearmanr(s, g)
    tau, tau_p  = kendalltau(s, g)

    topk = {}
    for pct in [0.01, 0.05, 0.10]:
        k        = max(1, int(n * pct))
        true_top = set(np.argsort(g)[-k:])
        pred_top = set(np.argsort(s)[-k:])
        topk[pct] = len(true_top & pred_top) / k

    def interp(r):
        a = abs(r)
        if a >= 0.6: return "Strong"
        if a >= 0.3: return "Moderate"
        if a > 0.0:  return "Weak"
        return "No signal"

    return {
        "proxy_name":          proxy_name,
        "count":               n,
        "spearman_rho":        float(rho),
        "spearman_pvalue":     float(rho_p),
        "kendall_tau":         float(tau),
        "kendall_pvalue":      float(tau_p),
        "interpretation":      interp(rho),
        "topk_precision_1pct": float(topk[0.01]),
        "topk_precision_5pct": float(topk[0.05]),
        "topk_precision_10pct":float(topk[0.10]),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_scatter(gt: np.ndarray, raw_arr: Dict[str, np.ndarray],
                 trans_arr: Dict[str, np.ndarray],
                 corrs_raw: dict, corrs_trans: dict):
    n_p   = len(PROXY_NAMES)
    fig, axes = plt.subplots(2, n_p, figsize=(5 * n_p, 9))
    fig.suptitle("NATS-Bench SSS — Proxy vs GT CIFAR-10 Accuracy", fontsize=13, fontweight="bold")

    for col, name in enumerate(PROXY_NAMES):
        disp = DISPLAY[name]

        # Row 0: raw scatter
        ax = axes[0, col]
        sc = raw_arr[name]
        mask = np.isfinite(sc) & np.isfinite(gt)
        ax.scatter(sc[mask], gt[mask], alpha=0.15, s=4, color="steelblue", rasterized=True)
        if mask.sum() > 1:
            z = np.polyfit(sc[mask], gt[mask], 1)
            xl = np.linspace(sc[mask].min(), sc[mask].max(), 200)
            ax.plot(xl, np.poly1d(z)(xl), "r--", lw=1.5, alpha=0.9)
        rho = corrs_raw.get(f"{name}_raw", {}).get("spearman_rho", float("nan"))
        ax.set_title(f"{disp} — Raw\nrho={rho:.4f}", fontsize=9, fontweight="bold")
        ax.set_xlabel("Score"); ax.set_ylabel("CIFAR-10 Acc (%)")

        # Row 1: rank-rank (transformed)
        ax2 = axes[1, col]
        sc2 = trans_arr.get(name)
        if sc2 is not None:
            mask2 = np.isfinite(sc2) & np.isfinite(gt)
            sr  = np.argsort(np.argsort(sc2[mask2]))
            gr  = np.argsort(np.argsort(gt[mask2]))
            ax2.scatter(sr, gr, alpha=0.1, s=4, color="darkorange", rasterized=True)
            z2 = np.polyfit(sr, gr, 1)
            ax2.plot(np.linspace(sr.min(), sr.max(), 200),
                     np.poly1d(z2)(np.linspace(sr.min(), sr.max(), 200)),
                     "r--", lw=1.5, alpha=0.9)
            rho2 = corrs_trans.get(f"{name}_transformed", {}).get("spearman_rho", float("nan"))
            ax2.set_title(f"{disp} — Rank (log-transformed)\nrho={rho2:.4f}", fontsize=9, fontweight="bold")
        else:
            ax2.set_visible(False)
        ax2.set_xlabel("Proxy rank"); ax2.set_ylabel("GT rank")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "scatter_plots.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'scatter_plots.png'}")


def plot_topk(corrs_raw: dict, corrs_trans: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("NATS-Bench SSS — Top-K Precision", fontsize=12)
    ks    = ["1%", "5%", "10%"]
    keys  = ["topk_precision_1pct", "topk_precision_5pct", "topk_precision_10pct"]
    x     = np.arange(len(ks))
    w     = 0.18
    colors = ["steelblue", "darkorange", "green", "purple"]

    for ax, (title, corrs, tag) in zip(axes, [
        ("Raw scores",         corrs_raw,   "_raw"),
        ("Log-transformed",    corrs_trans, "_transformed"),
    ]):
        for i, name in enumerate(PROXY_NAMES):
            key = f"{name}{tag}"
            if key not in corrs: continue
            vals = [corrs[key].get(k, 0) for k in keys]
            ax.bar(x + i * w, vals, width=w, label=DISPLAY[name], color=colors[i], alpha=0.8)
        ax.axhline(1.0, color="black", lw=0.8, linestyle="--", label="perfect")
        ax.set_title(title)
        ax.set_xticks(x + w * 1.5)
        ax.set_xticklabels([f"Top {k}" for k in ks])
        ax.set_ylabel("Precision")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "topk_precision.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'topk_precision.png'}")


def plot_gt_distribution(gt: np.ndarray):
    fig, ax = plt.subplots(figsize=(8, 4))
    valid = gt[np.isfinite(gt)]
    ax.hist(valid, bins=100, color="teal", edgecolor="none", alpha=0.8)
    ax.axvline(valid.mean(), color="crimson", lw=1.5, linestyle="--",
               label=f"mean={valid.mean():.2f}%")
    ax.axvline(np.median(valid), color="orange", lw=1.5, linestyle=":",
               label=f"median={np.median(valid):.2f}%")
    ax.set_xlabel("CIFAR-10 Accuracy (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"NATS-Bench SSS GT Accuracy Distribution  (N={len(valid):,})")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'accuracy_distribution.png'}")
    print(f"  GT stats: min={valid.min():.2f}  max={valid.max():.2f}"
          f"  mean={valid.mean():.2f}  std={valid.std():.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Step 4 -- Ranking Correlation Validation (NATS-Bench SSS)")
    print("=" * 60)

    # Load GT
    gt = load_gt_accuracy()

    # Load proxies
    raw_arr   = {}
    trans_arr = {}
    for name in PROXY_NAMES:
        raw_arr[name]   = load_proxy(name, "raw")
        trans_arr[name] = load_proxy(name, "transformed")
        print(f"  Loaded {name}: raw shape={raw_arr[name].shape}")

    # --- GT distribution plot ---
    print("\n[Plot] GT accuracy distribution ...")
    plot_gt_distribution(gt)

    # --- Compute correlations ---
    print("\n[Correlations] Raw proxies:")
    corrs_raw = {}
    for name in PROXY_NAMES:
        c = compute_correlations(raw_arr[name], gt, f"{name}_raw")
        corrs_raw[f"{name}_raw"] = c
        print(f"  {name:14}  rho={c['spearman_rho']:+.4f}  tau={c['kendall_tau']:+.4f}"
              f"  top1%={c['topk_precision_1pct']:.2f}"
              f"  top5%={c['topk_precision_5pct']:.2f}"
              f"  top10%={c['topk_precision_10pct']:.2f}"
              f"  [{c['interpretation']}]")

    print("\n[Correlations] Log-transformed proxies:")
    corrs_trans = {}
    for name in PROXY_NAMES:
        c = compute_correlations(trans_arr[name], gt, f"{name}_transformed")
        corrs_trans[f"{name}_transformed"] = c
        print(f"  {name:14}  rho={c['spearman_rho']:+.4f}  tau={c['kendall_tau']:+.4f}"
              f"  top1%={c['topk_precision_1pct']:.2f}"
              f"  top5%={c['topk_precision_5pct']:.2f}"
              f"  top10%={c['topk_precision_10pct']:.2f}"
              f"  [{c['interpretation']}]")

    # --- Save JSON ---
    all_corrs = {}
    all_corrs.update(corrs_raw)
    all_corrs.update(corrs_trans)
    out_json = OUT_DIR / "correlation_results.json"
    with open(out_json, "w") as f:
        json.dump(all_corrs, f, indent=2)
    print(f"\n  Saved: {out_json}")

    # --- Save CSV (optional, no pandas required) ---
    try:
        import pandas as pd
        rows = list(corrs_raw.values()) + list(corrs_trans.values())
        df = pd.DataFrame(rows)
        csv_path = OUT_DIR / "correlation_table.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
    except ImportError:
        pass

    # --- Plots ---
    print("\n[Plots] ...")
    plot_scatter(gt, raw_arr, trans_arr, corrs_raw, corrs_trans)
    plot_topk(corrs_raw, corrs_trans)

    print("\nDone.")


if __name__ == "__main__":
    main()
