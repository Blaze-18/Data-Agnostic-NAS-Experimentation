"""
Step 3 -- Distribution Analysis for Proxy Metrics (NATS-Bench SSS).

Loads both raw (.npy from raw_proxy_scores/) and log-transformed (.npy from
transformed_proxy/) arrays, computes statistical summaries, and produces
distribution plots.

SSS notes vs NAS-Bench-201:
  - N=32,768 architectures (vs 15,625)
  - No degenerate 'none'-op cluster expected; distributions should be smoother
  - param_count spans 5 discrete channel-width positions (8 values each)
    -> expect multi-modal but more spread than NAS-Bench-201

Outputs (results/nats_bench_sss/proxy_distribution/):
  stats_summary.json
  distribution_plots.png
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

RAW_DIR   = Path("results/nats_bench_sss/raw_proxy_scores")
TRANS_DIR = Path("results/nats_bench_sss/transformed_proxy")
OUT_DIR   = Path("results/nats_bench_sss/proxy_distribution")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROXY_NAMES = ["param_count", "naswot", "zenscore", "synflow"]
DISPLAY     = {
    "param_count": "Param Count",
    "naswot":      "NASWOT",
    "zenscore":    "Zen-Score",
    "synflow":     "SynFlow",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_stats(arr: np.ndarray, label: str) -> dict:
    v = arr[np.isfinite(arr)]
    near_zero = int(np.sum(v < 1e-6))
    return {
        "label":          label,
        "count":          int(len(v)),
        "mean":           float(v.mean()),
        "median":         float(np.median(v)),
        "std":            float(v.std()),
        "min":            float(v.min()),
        "max":            float(v.max()),
        "q25":            float(np.percentile(v, 25)),
        "q75":            float(np.percentile(v, 75)),
        "skewness":       float(np.mean(((v - v.mean()) / (v.std() + 1e-12)) ** 3)),
        "kurtosis":       float(np.mean(((v - v.mean()) / (v.std() + 1e-12)) ** 4)),
        "cv":             float(v.std() / (abs(v.mean()) + 1e-12)),
        "near_zero_pct":  float(near_zero / len(v) * 100),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Step 3 -- Distribution Analysis (NATS-Bench SSS)")
    print("=" * 60)

    raw_data   = {}
    trans_data = {}
    all_stats  = {}

    # ---- Load arrays -------------------------------------------------------
    print("\n[1/3] Loading proxy arrays ...")
    for name in PROXY_NAMES:
        rp = RAW_DIR  / f"{name}.npy"
        tp = TRANS_DIR / f"{name}.npy"
        if not rp.exists():
            print(f"  WARN  {name} raw not found: {rp}")
            continue
        raw_data[name]   = np.load(rp)
        trans_data[name] = np.load(tp) if tp.exists() else None

        s_raw   = compute_stats(raw_data[name],   f"{name}_raw")
        all_stats[f"{name}_raw"] = s_raw
        if trans_data[name] is not None:
            s_log = compute_stats(trans_data[name], f"{name}_log")
            all_stats[f"{name}_log"] = s_log

        print(f"  {name:14}  raw: mean={s_raw['mean']:.4e}  std={s_raw['std']:.4e}"
              f"  cv={s_raw['cv']:.3f}  near_zero={s_raw['near_zero_pct']:.2f}%")

    # ---- Save JSON ---------------------------------------------------------
    print("\n[2/3] Saving stats ...")
    out_json = OUT_DIR / "stats_summary.json"
    with open(out_json, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"  Saved: {out_json}")

    # ---- Plots -------------------------------------------------------------
    print("\n[3/3] Generating plots ...")
    n_proxies = len(raw_data)
    fig = plt.figure(figsize=(5 * n_proxies, 10))
    gs  = gridspec.GridSpec(3, n_proxies, hspace=0.45, wspace=0.3)

    for col, name in enumerate(PROXY_NAMES):
        if name not in raw_data:
            continue
        raw   = raw_data[name]
        trans = trans_data.get(name)
        disp  = DISPLAY[name]

        # Row 0: Raw histogram
        ax0 = fig.add_subplot(gs[0, col])
        ax0.hist(raw, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
        ax0.set_title(f"{disp}\nRaw", fontsize=9)
        ax0.set_xlabel("Score")
        ax0.set_ylabel("Count")
        s = all_stats[f"{name}_raw"]
        ax0.axvline(s["mean"],   color="crimson", lw=1.2, linestyle="--", label="mean")
        ax0.axvline(s["median"], color="orange",  lw=1.2, linestyle=":",  label="median")
        ax0.legend(fontsize=7, framealpha=0.6)

        # Row 1: Log-transformed histogram
        ax1 = fig.add_subplot(gs[1, col])
        if trans is not None:
            ax1.hist(trans, bins=80, color="darkorange", edgecolor="none", alpha=0.8)
            s2 = all_stats[f"{name}_log"]
            ax1.axvline(s2["mean"],   color="blue",   lw=1.2, linestyle="--")
            ax1.axvline(s2["median"], color="purple",  lw=1.2, linestyle=":")
            ax1.set_title(f"{disp}\nLog-transformed", fontsize=9)
        else:
            ax1.set_visible(False)
        ax1.set_xlabel("log(Score)")
        ax1.set_ylabel("Count")

        # Row 2: Box plot (raw)
        ax2 = fig.add_subplot(gs[2, col])
        ax2.boxplot(raw[np.isfinite(raw)], vert=True, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="steelblue"),
                    medianprops=dict(color="crimson", linewidth=2))
        ax2.set_title(f"{disp}\nBox (raw)", fontsize=9)
        ax2.set_ylabel("Score")
        ax2.set_xticks([])

    fig.suptitle("NATS-Bench SSS — Proxy Score Distributions (N=32,768)", fontsize=12, y=1.01)
    out_png = OUT_DIR / "distribution_plots.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_png}")

    # ---- Print summary table -----------------------------------------------
    print("\n  Summary (raw):")
    print(f"  {'proxy':14}  {'mean':>12}  {'std':>12}  {'cv':>6}  {'skew':>6}  {'kurt':>6}")
    print("  " + "-" * 65)
    for name in PROXY_NAMES:
        if name not in raw_data:
            continue
        s = all_stats[f"{name}_raw"]
        print(f"  {name:14}  {s['mean']:12.4e}  {s['std']:12.4e}  {s['cv']:6.3f}"
              f"  {s['skewness']:6.3f}  {s['kurtosis']:6.3f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
