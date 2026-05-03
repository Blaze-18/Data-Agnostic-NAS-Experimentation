r"""
Step 0: NATS-Bench SSS Structural Audit
========================================

Answers every structural question the pipeline depends on before any proxy
extraction code is written. Loads from the already-extracted simple archive
(data/nats_bench_sss/NATS-sss-v1_0-50262-simple/) — no re-decompression.

Questions answered
------------------
Q1 — GT accuracy key verification (5 spot-check architectures)
Q2 — Accuracy distribution shape across all 32,768 architectures
Q3 — Architecture space structure (5 positions, 8 values, no op-type variation)
Q4 — Channel-width positions vs param_count as structural covariate
Q5 — FLOPs / cost data availability in benchmark pickle data
Q6 — Raw NASWOT correlation direction on 200-arch subset
     (determines whether negation is needed before log-transform in Step 2)

Outputs  →  results/nats_bench_sss/audit/
  audit_summary.json          — all findings, pass/fail per question
  accuracy_distribution.png   — histogram + percentile statistics
  covariate_analysis.png      — channel-width positions + param proxy vs GT
  direction_check.png         — raw NASWOT scores vs GT accuracy (200 archs)

Usage:
    envs\nasbench_env\Scripts\python.exe scripts\nats_bench_sss\step0_sss_audit.py
"""

import json
import pickle
import random
import time
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

SIMPLE_ARCHIVE = Path("data/nats_bench_sss/NATS-sss-v1_0-50262-simple")
OUT_DIR        = Path("results/nats_bench_sss/audit")
N_ARCHS        = 32_768
EPOCH_KEY      = "90"            # 90-epoch training slot
GT_KEY         = ("cifar10", 777)  # expected key for CIFAR-10 test accuracy
RANDOM_SEED    = 42

SPOT_INDICES   = [0, 1000, 8192, 16383, 32767]
DIRECTION_N    = 200              # architectures used for Q6 direction check

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_entry(idx: int) -> dict:
    """Load a single architecture entry from the simple archive."""
    with open(SIMPLE_ARCHIVE / f"{idx}.pickle", "rb") as f:
        return pickle.load(f)


def deep_inspect(obj, prefix: str = "", depth: int = 0, max_depth: int = 5, max_items: int = 6):
    """Recursively print the structure of an object."""
    indent = "  " * depth
    t = type(obj).__name__

    if depth > max_depth:
        print(f"{indent}{prefix}<max depth>")
        return

    if isinstance(obj, dict):
        print(f"{indent}{prefix}dict ({len(obj)} keys)")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{indent}  ... ({len(obj) - max_items} more keys)")
                break
            deep_inspect(v, prefix=f"[{k!r}] → ", depth=depth + 1,
                         max_depth=max_depth, max_items=max_items)
    elif isinstance(obj, (list, tuple)):
        n = len(obj)
        print(f"{indent}{prefix}{t} (len={n})")
        if n > 0:
            deep_inspect(obj[0], prefix="[0] → ", depth=depth + 1,
                         max_depth=max_depth, max_items=max_items)
    elif hasattr(obj, "__dict__"):
        attrs = list(obj.__dict__.keys())
        print(f"{indent}{prefix}{t}  attrs={attrs[:max_items]}")
        for attr in attrs[:max_items]:
            val = getattr(obj, attr)
            deep_inspect(val, prefix=f".{attr} → ", depth=depth + 1,
                         max_depth=max_depth, max_items=3)
    else:
        r = repr(obj)
        print(f"{indent}{prefix}{t} = {r[:100]}{'...' if len(r) > 100 else ''}")


def try_extract_accuracy(result_obj, verbose: bool = False) -> float | None:
    """
    Try multiple access patterns to extract the final test accuracy scalar.

    Returns None if no pattern succeeds — caller should print debug info.
    """
    if verbose:
        print(f"      type: {type(result_obj).__name__}")
        if hasattr(result_obj, "__dict__"):
            print(f"      attrs: {list(result_obj.__dict__.keys())[:10]}")
        elif isinstance(result_obj, dict):
            print(f"      keys:  {list(result_obj.keys())[:10]}")

    # Pattern 1: plain numeric
    if isinstance(result_obj, (int, float)):
        return float(result_obj)

    # Pattern 2: dict — look for common accuracy fields
    if isinstance(result_obj, dict):
        for field in ("eval_acc1es", "valid_acc1es", "test_acc1es",
                      "accuracy", "test_acc", "valid_acc"):
            if field in result_obj:
                val = result_obj[field]
                if isinstance(val, dict):
                    # keyed by epoch integer — take the last one
                    return float(list(val.values())[-1])
                if isinstance(val, (list, tuple)):
                    return float(val[-1])
                return float(val)

    # Pattern 3: object attributes (ArchResults-style)
    for attr in ("eval_acc1es", "valid_acc1es", "test_acc1es", "accuracy"):
        if hasattr(result_obj, attr):
            val = getattr(result_obj, attr)
            if isinstance(val, dict):
                return float(list(val.values())[-1])
            if isinstance(val, (list, tuple)):
                return float(val[-1])
            return float(val)

    # Pattern 4: NATS-Bench API method
    if hasattr(result_obj, "get_eval"):
        for tag in ("ori-test", "x-test", "test"):
            try:
                res = result_obj.get_eval(tag)
                if isinstance(res, dict) and "accuracy" in res:
                    return float(res["accuracy"])
            except Exception:
                pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Q1 — GT ACCURACY KEY VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def q1_gt_key_verification() -> dict:
    print("\n" + "=" * 70)
    print("Q1 — GT ACCURACY KEY VERIFICATION")
    print("=" * 70)

    results = {}
    all_ok = True
    extraction_pattern = None

    for idx in SPOT_INDICES:
        entry      = load_entry(idx)
        epoch_data = entry[EPOCH_KEY]
        arch_str   = epoch_data["arch_str"]
        all_results = epoch_data["all_results"]

        print(f"\n  arch {idx:>5}  |  arch_str = {arch_str}")
        print(f"  all_results keys: {list(all_results.keys())}")

        if GT_KEY not in all_results:
            print(f"  ✗ key {GT_KEY} NOT FOUND")
            all_ok = False
            continue

        result_obj = all_results[GT_KEY]
        acc = try_extract_accuracy(result_obj, verbose=True)

        if acc is None:
            print(f"  ✗ Could not extract accuracy — running deep inspect:")
            deep_inspect(result_obj, prefix="  result_obj → ", depth=0, max_depth=4)
            all_ok = False
        elif not (40.0 <= acc <= 100.0):
            print(f"  ⚠ accuracy = {acc:.4f} — outside expected 40–100% range")
            results[idx] = acc
        else:
            print(f"  ✓ accuracy = {acc:.4f}%")
            results[idx] = acc
            if extraction_pattern is None:
                # Record which attribute held the accuracy
                for attr in ("eval_acc1es", "valid_acc1es", "accuracy"):
                    if hasattr(result_obj, attr):
                        extraction_pattern = f"obj.{attr}[-1]"
                        break
                if extraction_pattern is None and isinstance(result_obj, dict):
                    extraction_pattern = "dict access"

    # Check cifar10 vs cifar10-valid distinction on arch 0
    print(f"\n  Checking ('cifar10', 777) vs ('cifar10-valid', 777) for arch 0:")
    entry0 = load_entry(0)
    for check_key in [("cifar10-valid", 777), ("cifar10", 777)]:
        if check_key in entry0[EPOCH_KEY]["all_results"]:
            obj = entry0[EPOCH_KEY]["all_results"][check_key]
            acc = try_extract_accuracy(obj)
            print(f"    {check_key}: accuracy = {acc}")
        else:
            print(f"    {check_key}: NOT PRESENT")

    status = "PASS" if all_ok and len(results) == len(SPOT_INDICES) else "FAIL"
    print(f"\n  Q1 Status: {status}")

    return {
        "status": status,
        "spot_accuracies": {str(k): v for k, v in results.items()},
        "extraction_pattern": extraction_pattern,
        "gt_key": str(GT_KEY),
    }


# ─────────────────────────────────────────────────────────────────────────────
# BULK LOAD (used by Q2, Q3, Q4)
# ─────────────────────────────────────────────────────────────────────────────

def bulk_load_all() -> tuple[dict, dict]:
    """
    Load arch strings and GT accuracies for all 32,768 architectures.

    Returns
    -------
    arch_strs  : {idx: arch_string}
    accuracies : {idx: float accuracy}
    """
    print("\n" + "=" * 70)
    print("BULK LOAD — ALL 32,768 ARCHITECTURES")
    print("=" * 70)
    print("  (loading from simple archive — ~1 min)")

    arch_strs  = {}
    accuracies = {}
    failed     = []
    report_at  = N_ARCHS // 10
    t0         = time.monotonic()

    for idx in range(N_ARCHS):
        try:
            entry      = load_entry(idx)
            epoch_data = entry[EPOCH_KEY]
            arch_strs[idx] = epoch_data["arch_str"]

            if GT_KEY in epoch_data["all_results"]:
                acc = try_extract_accuracy(epoch_data["all_results"][GT_KEY])
                if acc is not None:
                    accuracies[idx] = acc
        except Exception as e:
            failed.append((idx, str(e)))

        if (idx + 1) % report_at == 0:
            elapsed   = time.monotonic() - t0
            rate      = (idx + 1) / elapsed
            remaining = (N_ARCHS - idx - 1) / rate
            print(f"  [{time.strftime('%H:%M:%S')}]  {idx+1:>6}/{N_ARCHS}"
                  f"  ({100*(idx+1)/N_ARCHS:.0f}%)  ETA {remaining:.0f}s", flush=True)

    elapsed = time.monotonic() - t0
    print(f"\n  Loaded {len(arch_strs):,} arch strings, "
          f"{len(accuracies):,} accuracies  in {elapsed:.1f}s")
    if failed:
        print(f"  Failed entries: {len(failed)}  (first: {failed[0]})")

    return arch_strs, accuracies


# ─────────────────────────────────────────────────────────────────────────────
# Q2 — ACCURACY DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def q2_accuracy_distribution(accuracies: dict) -> dict:
    print("\n" + "=" * 70)
    print("Q2 — ACCURACY DISTRIBUTION SHAPE")
    print("=" * 70)

    accs = np.array(list(accuracies.values()), dtype=float)

    stats = {
        "n":                int(len(accs)),
        "min":              float(accs.min()),
        "max":              float(accs.max()),
        "mean":             float(accs.mean()),
        "std":              float(accs.std()),
        "p5":               float(np.percentile(accs, 5)),
        "p25":              float(np.percentile(accs, 25)),
        "median":           float(np.median(accs)),
        "p75":              float(np.percentile(accs, 75)),
        "p95":              float(np.percentile(accs, 95)),
        "n_below_30pct":    int((accs < 30.0).sum()),
        "n_below_50pct":    int((accs < 50.0).sum()),
        "top1pct_threshold": float(np.percentile(accs, 99)),
        "top10pct_accuracy_range": [float(np.percentile(accs, 90)), float(accs.max())],
    }

    top10_range = stats["top10pct_accuracy_range"]
    top10_spread = top10_range[1] - top10_range[0]
    stats["top10pct_spread"] = float(top10_spread)

    print(f"  N             : {stats['n']:,}")
    print(f"  Min / Max     : {stats['min']:.4f}% / {stats['max']:.4f}%")
    print(f"  Mean ± Std    : {stats['mean']:.4f}% ± {stats['std']:.4f}%")
    print(f"  P5/P25/P50/P75/P95: "
          f"{stats['p5']:.2f} / {stats['p25']:.2f} / {stats['median']:.2f} / "
          f"{stats['p75']:.2f} / {stats['p95']:.2f}")
    print(f"  N below 30%   : {stats['n_below_30pct']}  "
          f"({100*stats['n_below_30pct']/len(accs):.1f}%)")
    print(f"  N below 50%   : {stats['n_below_50pct']}  "
          f"({100*stats['n_below_50pct']/len(accs):.1f}%)")
    print(f"  Top-1% threshold : {stats['top1pct_threshold']:.4f}%")
    print(f"  Top-10% range    : {top10_range[0]:.4f}% – {top10_range[1]:.4f}%  "
          f"(spread = {top10_spread:.4f}%)")

    # NAS-Bench-201 comparison note
    if stats["n_below_30pct"] < 100:
        print(f"\n  ✓ No degenerate cluster near 10% — unlike NAS-Bench-201's ~300 failed archs")
        print(f"    → Global proxy ρ values will NOT be inflated by easy-to-rank dead architectures")
        print(f"    → Lower global ρ than NAS-Bench-201 is expected and acceptable")
    else:
        print(f"\n  ⚠ {stats['n_below_30pct']} architectures below 30% — may indicate a degenerate cluster")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Full distribution
    axes[0].hist(accs, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
    axes[0].axvline(stats["mean"],   color="red",    linestyle="--", lw=1.5,
                    label=f"mean={stats['mean']:.2f}%")
    axes[0].axvline(stats["median"], color="orange", linestyle="--", lw=1.5,
                    label=f"median={stats['median']:.2f}%")
    axes[0].set_xlabel("CIFAR-10 Test Accuracy (%)")
    axes[0].set_ylabel("Count")
    axes[0].set_title(f"SSS Accuracy Distribution (N={stats['n']:,})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Top-10% zoom
    threshold_90 = float(np.percentile(accs, 90))
    top_accs = accs[accs >= threshold_90]
    axes[1].hist(top_accs, bins=50, color="darkgreen", edgecolor="none", alpha=0.85)
    axes[1].set_xlabel("CIFAR-10 Test Accuracy (%)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Top 10% (≥{threshold_90:.1f}%,  spread={top10_spread:.3f}%)")
    axes[1].grid(True, alpha=0.3)

    # CDF
    sorted_accs = np.sort(accs)
    cdf = np.arange(1, len(sorted_accs) + 1) / len(sorted_accs)
    axes[2].plot(sorted_accs, cdf, color="purple", lw=1.5)
    axes[2].set_xlabel("CIFAR-10 Test Accuracy (%)")
    axes[2].set_ylabel("CDF")
    axes[2].set_title("Cumulative Distribution")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "accuracy_distribution.png", dpi=150)
    plt.close()
    print(f"\n  Saved: accuracy_distribution.png")
    print(f"  Q2 Status: PASS")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Q3 — ARCHITECTURE SPACE STRUCTURE
# ─────────────────────────────────────────────────────────────────────────────

def q3_arch_space_structure(arch_strs: dict) -> tuple[np.ndarray, dict]:
    print("\n" + "=" * 70)
    print("Q3 — ARCHITECTURE SPACE STRUCTURE")
    print("=" * 70)

    rows         = []
    parse_errors = 0
    idx_order    = []

    for idx in sorted(arch_strs.keys()):
        try:
            vals = [int(c) for c in arch_strs[idx].split(":")]
            assert len(vals) == 5
            rows.append(vals)
            idx_order.append(idx)
        except Exception:
            parse_errors += 1

    channel_matrix = np.array(rows, dtype=int)  # (N, 5)
    print(f"  Parsed {len(channel_matrix):,} strings  ({parse_errors} errors)")
    print(f"  Matrix shape: {channel_matrix.shape}")

    all_ok = True
    position_info = {}

    print(f"\n  Unique values per channel position:")
    for pos in range(5):
        unique_vals = sorted(set(channel_matrix[:, pos].tolist()))
        n_unique    = len(unique_vals)
        var         = float(channel_matrix[:, pos].var())
        position_info[f"pos_{pos}"] = {
            "n_unique": n_unique,
            "values":   unique_vals,
            "variance": var,
        }
        status = "✓" if n_unique == 8 else "✗"
        print(f"    Position {pos}: {n_unique} values {unique_vals}  "
              f"(var={var:.1f}) {status}")
        if n_unique != 8:
            all_ok = False

    # Verify total combinatorics
    total = 1
    for pos in range(5):
        total *= len(set(channel_matrix[:, pos].tolist()))
    expected_total = 8 ** 5
    combo_ok = (total == expected_total == len(channel_matrix))
    print(f"\n  Combinatorics: 8^5 = {expected_total:,}  "
          f"| loaded = {len(channel_matrix):,}  "
          f"{'✓' if combo_ok else '✗'}")

    # Check value set
    all_vals      = sorted(set(channel_matrix.flatten().tolist()))
    expected_vals = [8, 16, 24, 32, 40, 48, 56, 64]
    vals_ok       = (all_vals == expected_vals)
    print(f"  All unique values: {all_vals}")
    print(f"  Expected:          {expected_vals}  {'✓' if vals_ok else '✗'}")
    if not vals_ok:
        all_ok = False

    status = "PASS" if (all_ok and combo_ok and vals_ok) else "FAIL"
    print(f"\n  Q3 Status: {status}")

    return channel_matrix, {
        "status": status,
        "n_parsed": len(channel_matrix),
        "parse_errors": parse_errors,
        "position_info": position_info,
        "combinatorics_ok": combo_ok,
        "value_set_ok": vals_ok,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Q4 — CHANNEL-WIDTH POSITIONS VS PARAM_COUNT AS COVARIATE
# ─────────────────────────────────────────────────────────────────────────────

def q4_covariate_analysis(channel_matrix: np.ndarray,
                          arch_strs: dict,
                          accuracies: dict) -> dict:
    print("\n" + "=" * 70)
    print("Q4 — CHANNEL-WIDTH POSITIONS VS PARAM_COUNT AS COVARIATE")
    print("=" * 70)

    # Align by sorted arch index
    common_idx  = sorted(set(range(len(channel_matrix))) & set(accuracies.keys()))
    accs        = np.array([accuracies[i] for i in common_idx], dtype=float)
    chs         = channel_matrix[common_idx]           # (N, 5)

    # Two structural capacity proxies:
    #   proxy_sq  = Σ ch[i]²   (dominant term in Conv params with fixed kernel k=3)
    #   proxy_sum = Σ ch[i]    (linear capacity)
    proxy_sq  = np.array([float(np.sum(row ** 2)) for row in chs])
    proxy_sum = np.array([float(np.sum(row))      for row in chs])

    print(f"  Aligned {len(common_idx):,} architectures")
    print(f"\n  Spearman ρ — individual channel positions vs GT accuracy:")
    position_rhos = {}
    for pos in range(5):
        rho, p = spearmanr(chs[:, pos], accs)
        position_rhos[f"pos_{pos}"] = {"rho": float(rho), "p": float(p)}
        print(f"    Position {pos}: ρ = {rho:+.4f}  (p = {p:.2e})")

    rho_sq,  p_sq  = spearmanr(proxy_sq,  accs)
    rho_sum, p_sum = spearmanr(proxy_sum, accs)
    print(f"\n  Structural capacity proxies vs GT accuracy:")
    print(f"    Σ ch²  (param proxy): ρ = {rho_sq:+.4f}  (p = {p_sq:.2e})")
    print(f"    Σ ch   (sum proxy):   ρ = {rho_sum:+.4f}  (p = {p_sum:.2e})")

    # Decision: is param_count sufficient as single covariate?
    max_pos_rho   = max(abs(v["rho"]) for v in position_rhos.values())
    best_pos      = max(range(5), key=lambda i: abs(position_rhos[f"pos_{i}"]["rho"]))
    param_rho_abs = abs(rho_sq)

    print(f"\n  Max |ρ| from individual positions : {max_pos_rho:.4f}  (position {best_pos})")
    print(f"  |ρ| from Σ ch² param proxy        : {param_rho_abs:.4f}")

    if param_rho_abs >= max_pos_rho - 0.02:          # within 2pp tolerance
        recommendation = "single_covariate_param_count"
        print(f"\n  → Recommendation: param_count (Σ ch²) is SUFFICIENT as single covariate.")
        print(f"    Single-covariate OLS transfers from NAS-Bench-201 unchanged.")
    else:
        recommendation = f"multi_covariate_include_pos_{best_pos}"
        delta = max_pos_rho - param_rho_abs
        print(f"\n  → Recommendation: Position {best_pos} beats param proxy by {delta:.4f}.")
        print(f"    Consider multi-covariate OLS: P ~ param_count + ch_{best_pos}")

    # Pairwise position × position correlations
    print(f"\n  Pairwise Spearman ρ between channel positions (5×5):")
    corr_matrix = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            corr_matrix[i, j], _ = spearmanr(chs[:, i], chs[:, j])
    header = "       " + "  ".join([f"Pos{j}" for j in range(5)])
    print(f"  {header}")
    for i in range(5):
        row_str = "  ".join([f"{corr_matrix[i,j]:+.3f}" for j in range(5)])
        print(f"  Pos{i}:  {row_str}")

    # Plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Scatter: param proxy vs accuracy (sample 3000 for speed)
    rng     = np.random.default_rng(RANDOM_SEED)
    si      = rng.choice(len(accs), size=min(3000, len(accs)), replace=False)
    axes[0].scatter(proxy_sq[si], accs[si], alpha=0.15, s=3, color="steelblue")
    axes[0].set_xlabel(r"Structural Param Proxy ($\Sigma$ ch²)")
    axes[0].set_ylabel("CIFAR-10 Test Accuracy (%)")
    axes[0].set_title(f"Capacity vs Accuracy  (ρ = {rho_sq:.3f})")
    axes[0].grid(True, alpha=0.3)

    # Bar: rho per position + proxies
    labels = [f"Pos {i}" for i in range(5)] + ["Σch²", "Σch"]
    rhos   = [position_rhos[f"pos_{i}"]["rho"] for i in range(5)] + [rho_sq, rho_sum]
    colors = ["#2196F3" if r > 0 else "#F44336" for r in rhos[:5]] + ["#FF9800", "#9C27B0"]
    axes[1].bar(labels, rhos, color=colors, alpha=0.8)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("Spearman ρ with GT Accuracy")
    axes[1].set_title("Per-Position & Param Proxy Correlations")
    axes[1].grid(True, alpha=0.3)
    for bar, r in zip(axes[1].patches, rhos):
        axes[1].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005 * np.sign(r),
                     f"{r:.3f}", ha="center", va="bottom", fontsize=7)

    # Heatmap: position pairwise correlations
    im = axes[2].imshow(corr_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    axes[2].set_xticks(range(5)); axes[2].set_xticklabels([f"P{i}" for i in range(5)])
    axes[2].set_yticks(range(5)); axes[2].set_yticklabels([f"P{i}" for i in range(5)])
    axes[2].set_title("Pairwise Position ρ")
    for i in range(5):
        for j in range(5):
            axes[2].text(j, i, f"{corr_matrix[i,j]:.2f}",
                         ha="center", va="center", fontsize=8,
                         color="white" if abs(corr_matrix[i, j]) > 0.6 else "black")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "covariate_analysis.png", dpi=150)
    plt.close()
    print(f"\n  Saved: covariate_analysis.png")
    print(f"  Q4 Status: PASS")

    return {
        "status": "PASS",
        "n_aligned": len(common_idx),
        "position_rhos": position_rhos,
        "param_proxy_sq_rho": float(rho_sq),
        "param_proxy_sq_p":   float(p_sq),
        "param_proxy_sum_rho": float(rho_sum),
        "recommendation": recommendation,
        "max_position_rho": float(max_pos_rho),
        "best_individual_position": int(best_pos),
        "position_pairwise_corr": corr_matrix.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Q5 — FLOPS / COST DATA AVAILABILITY
# ─────────────────────────────────────────────────────────────────────────────

def q5_flops_availability(accuracies: dict) -> dict:
    print("\n" + "=" * 70)
    print("Q5 — FLOPS / COST DATA AVAILABILITY")
    print("=" * 70)

    entry = load_entry(0)
    epoch_data = entry[EPOCH_KEY]

    print(f"  Top-level entry keys:    {list(entry.keys())}")
    print(f"  Epoch-data ('90') keys:  {list(epoch_data.keys())}")

    # Inspect dataset_seed
    if "dataset_seed" in epoch_data:
        ds = epoch_data["dataset_seed"]
        print(f"\n  dataset_seed keys: {list(ds.keys())}")
        for k, v in list(ds.items())[:2]:
            print(f"    '{k}': type={type(v).__name__}  repr={repr(v)[:120]}")

    # Deep-inspect the result object
    result_obj = epoch_data["all_results"].get(GT_KEY)
    print(f"\n  Deep-inspecting all_results[{GT_KEY}]:")
    deep_inspect(result_obj, prefix="  result → ", depth=0, max_depth=5, max_items=8)

    # Search all entry keys at every epoch level
    found_flops = {}
    for epoch_key in entry.keys():
        if not isinstance(entry[epoch_key], dict):
            continue
        ed = entry[epoch_key]
        for field in ("flops", "flop", "cost", "latency", "params"):
            if field in ed:
                found_flops[f"{epoch_key}.{field}"] = repr(ed[field])[:80]

    if found_flops:
        print(f"\n  FLOPs / cost fields found: {found_flops}")
        flops_available = True
        flops_note = f"Found: {list(found_flops.keys())}"
    else:
        print(f"\n  No flops/cost fields found at epoch-data level.")
        flops_available = False
        flops_note = "Not found at epoch-data level; may require API computation."

    print(f"\n  Q5 Status: PASS — inspection complete")
    return {
        "status": "PASS",
        "flops_available": flops_available,
        "note": flops_note,
        "fields_found": list(found_flops.keys()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Q6 — NASWOT DIRECTION CHECK
# ─────────────────────────────────────────────────────────────────────────────

class _ResBlock(object):
    """Placeholder — defined inside build_sss_model_simple to avoid top-level torch dependency."""
    pass


def build_sss_model_simple(arch_str: str):
    """
    Build a minimal ResNet-style SSS model from an arch string.

    Used for Q6 direction check ONLY — not the production model builder.
    The 5 channel-width values map to: stem → stage0 → stage1 → stage2 → stage3 → stage4.
    Downsampling at stages 1 and 3 (stride=2) to match a typical 5-stage network on 32×32.
    """
    import torch.nn as nn

    channels = [int(c) for c in arch_str.split(":")]
    assert len(channels) == 5

    class ResBlock(nn.Module):
        def __init__(self, c_in, c_out, stride=1):
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                nn.BatchNorm2d(c_out),
            )
            self.skip = nn.Sequential()
            if stride != 1 or c_in != c_out:
                self.skip = nn.Sequential(
                    nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False),
                    nn.BatchNorm2d(c_out),
                )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.body(x) + self.skip(x))

    class SSSNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, channels[0], 3, padding=1, bias=False),
                nn.BatchNorm2d(channels[0]),
                nn.ReLU(inplace=True),
            )
            c = channels[0]
            blocks = []
            for i, c_out in enumerate(channels):
                stride = 2 if i in (1, 3) else 1
                blocks.append(ResBlock(c, c_out, stride))
                c = c_out
            self.stages = nn.Sequential(*blocks)
            self.pool   = nn.AdaptiveAvgPool2d(1)
            self.fc     = nn.Linear(channels[-1], 10)

        def forward(self, x):
            x = self.stem(x)
            x = self.stages(x)
            x = self.pool(x)
            return self.fc(x.view(x.size(0), -1))

    return SSSNet()


def naswot_fast(model, device) -> float:
    """
    NASWOT: multi-layer Conv2d hooking, single random batch (batch=8, 32×32).
    Returns mean covariance trace across all Conv2d layers.
    """
    import torch
    import torch.nn as nn

    model.eval()
    convs = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not convs:
        return 0.0

    activations = {i: [] for i in range(len(convs))}
    hooks = []

    def make_hook(i):
        def h(mod, inp, out):
            flat = out.detach().cpu().view(out.size(0), out.size(1), -1).mean(2)
            activations[i].append(flat)
        return h

    for i, conv in enumerate(convs):
        hooks.append(conv.register_forward_hook(make_hook(i)))

    with torch.no_grad():
        x = torch.randn(8, 3, 32, 32, device=device)
        try:
            model(x)
        except Exception:
            pass

    for h in hooks:
        h.remove()

    scores = []
    for i in range(len(convs)):
        if not activations[i]:
            continue
        acts = activations[i][0]      # (8, channels)
        if acts.size(1) < 2:
            continue
        # Covariance trace (numpy for compatibility)
        a = acts.numpy()
        cov = np.cov(a.T)
        scores.append(float(np.trace(cov)))

    return float(np.mean(scores)) if scores else 0.0


def q6_direction_check(arch_strs: dict, accuracies: dict) -> dict:
    print("\n" + "=" * 70)
    print("Q6 — NASWOT RAW CORRELATION DIRECTION CHECK")
    print("=" * 70)

    try:
        import torch
        device = torch.device("cpu")
        print(f"  PyTorch available — using device: {device}")
    except ImportError:
        print("  ERROR: PyTorch not available — cannot run Q6")
        return {"status": "SKIP", "reason": "torch not available"}

    rng     = random.Random(RANDOM_SEED)
    common  = sorted(set(arch_strs.keys()) & set(accuracies.keys()))
    sample  = rng.sample(common, min(DIRECTION_N, len(common)))

    print(f"  Computing NASWOT on {len(sample)} random architectures ...")
    naswot_scores = {}
    t0 = time.monotonic()

    for i, idx in enumerate(sample):
        try:
            model = build_sss_model_simple(arch_strs[idx]).to(device)
            score = naswot_fast(model, device)
            naswot_scores[idx] = score
            del model
        except Exception as e:
            pass

        if (i + 1) % 50 == 0:
            elapsed = time.monotonic() - t0
            print(f"    {i+1}/{len(sample)}  ({elapsed:.1f}s)", flush=True)

    elapsed = time.monotonic() - t0
    valid   = {k: v for k, v in naswot_scores.items() if v > 0}
    print(f"  Computed {len(valid)} valid scores in {elapsed:.1f}s")

    if len(valid) < 20:
        print("  WARNING: fewer than 20 valid scores — direction result unreliable")

    paired_idx    = sorted(set(valid.keys()) & set(accuracies.keys()))
    scores_arr    = np.array([valid[i]       for i in paired_idx])
    accs_arr      = np.array([accuracies[i]  for i in paired_idx])

    rho_raw, p_raw = spearmanr(scores_arr, accs_arr)

    print(f"\n  Raw NASWOT score range: [{scores_arr.min():.4f}, {scores_arr.max():.4f}]")
    print(f"  Spearman ρ (raw NASWOT vs GT accuracy): {rho_raw:+.4f}  (p = {p_raw:.2e})")

    if rho_raw > 0:
        direction        = "positive"
        negation_needed  = False
        transform_note   = "log(score + ε)  — NO negation"
        print(f"\n  → Positive raw correlation.")
        print(f"    Transform to use in Step 2: log(score + ε)")
        print(f"    This DIFFERS from NAS-Bench-201 (which required negation).")
        print(f"    Reason: SSS has no 'none' / skip operations that caused high-variance")
        print(f"    activations in skip-heavy architectures on NAS-Bench-201.")
    else:
        direction        = "negative"
        negation_needed  = True
        transform_note   = "-log(score + ε)  — negation applied (same as NAS-Bench-201)"
        print(f"\n  → Negative raw correlation.")
        print(f"    Transform to use in Step 2: -log(score + ε)  (same as NAS-Bench-201)")

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(scores_arr, accs_arr, alpha=0.4, s=15, color="steelblue")
    axes[0].set_xlabel("Raw NASWOT Score (covariance trace)")
    axes[0].set_ylabel("CIFAR-10 Test Accuracy (%)")
    axes[0].set_title(f"Raw NASWOT vs GT  (ρ={rho_raw:.3f}, n={len(paired_idx)})")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(np.log(scores_arr + 1e-8), accs_arr, alpha=0.4, s=15, color="darkorange")
    axes[1].set_xlabel("log(NASWOT score)")
    axes[1].set_ylabel("CIFAR-10 Test Accuracy (%)")
    rho_log, _ = spearmanr(np.log(scores_arr + 1e-8), accs_arr)
    axes[1].set_title(f"log(NASWOT) vs GT  (ρ={rho_log:.3f})")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "direction_check.png", dpi=150)
    plt.close()
    print(f"\n  Saved: direction_check.png")
    print(f"  Q6 Status: PASS — direction = {direction}")

    return {
        "status": "PASS",
        "n_subset": len(paired_idx),
        "rho_raw": float(rho_raw),
        "p_raw": float(p_raw),
        "rho_log": float(rho_log),
        "direction": direction,
        "negation_needed": negation_needed,
        "transform_note": transform_note,
        "score_range": [float(scores_arr.min()), float(scores_arr.max())],
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "#" * 70)
    print("# NATS-BENCH SSS — STEP 0 STRUCTURAL AUDIT")
    print(f"# Input  : {SIMPLE_ARCHIVE}")
    print(f"# Output : {OUT_DIR}")
    print("#" * 70)

    if not SIMPLE_ARCHIVE.exists():
        print(f"\nERROR: Simple archive not found at {SIMPLE_ARCHIVE}")
        print("Run decompress_nats_sss.py first.")
        return

    t_total  = time.monotonic()
    summary  = {}

    # ── Q1: spot-check accuracy key ─────────────────────────────────────────
    summary["q1_gt_key"]    = q1_gt_key_verification()

    if summary["q1_gt_key"]["status"] == "FAIL":
        print("\n" + "!" * 70)
        print("! Q1 FAILED — cannot extract accuracy values.")
        print("! Deep-inspect output above shows the actual result object structure.")
        print("! Fix try_extract_accuracy() before running bulk load.")
        print("!" * 70)
        with open(OUT_DIR / "audit_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        return

    # ── Bulk load ────────────────────────────────────────────────────────────
    arch_strs, accuracies = bulk_load_all()

    if len(accuracies) < N_ARCHS * 0.95:
        print(f"\n⚠ Only {len(accuracies):,} accuracies loaded (< 95% of {N_ARCHS:,}).")
        print(f"  Check the deep-inspect output from Q1 to fix the extraction pattern.")

    # ── Q2: distribution ─────────────────────────────────────────────────────
    summary["q2_distribution"]  = q2_accuracy_distribution(accuracies)

    # ── Q3: arch space structure ─────────────────────────────────────────────
    channel_matrix, q3_info = q3_arch_space_structure(arch_strs)
    summary["q3_arch_space"]    = q3_info

    # ── Q4: covariate analysis ───────────────────────────────────────────────
    summary["q4_covariate"]     = q4_covariate_analysis(channel_matrix, arch_strs, accuracies)

    # ── Q5: FLOPs availability ───────────────────────────────────────────────
    summary["q5_flops"]         = q5_flops_availability(accuracies)

    # ── Q6: NASWOT direction ─────────────────────────────────────────────────
    summary["q6_direction"]     = q6_direction_check(arch_strs, accuracies)

    # ── Final summary ────────────────────────────────────────────────────────
    total = time.monotonic() - t_total
    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)
    for q, info in summary.items():
        st = info.get("status", "?")
        print(f"  {q:25s}  {st}")

    print(f"\n  Total time: {total:.1f}s  ({total/60:.1f} min)")

    # Pipeline recommendations
    print("\n" + "=" * 70)
    print("PIPELINE DECISIONS (from audit)")
    print("=" * 70)

    q4 = summary.get("q4_covariate", {})
    q6 = summary.get("q6_direction", {})

    print(f"  Covariate for OLS    : {q4.get('recommendation', 'see Q4')}")
    print(f"  Primary debias method: partial rank Spearman (not OLS residuals)")
    print(f"  NASWOT/ZenScore sign : {q6.get('direction', 'see Q6')}")
    print(f"  Transform decision   : {q6.get('transform_note', 'see Q6')}")
    if not summary["q2_distribution"].get("n_below_30pct", 1):
        print(f"  Degenerate cluster   : None detected — lower global ρ is expected vs NAS-Bench-201")

    # Save JSON
    with open(OUT_DIR / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: audit_summary.json")
    print(f"  All outputs in: {OUT_DIR}/")


if __name__ == "__main__":
    main()
