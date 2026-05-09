"""
Step 0: Audit NAS-Bench-101 -- extract GT accuracies, param counts, and arch specs.

Loads the full TFRecord ONCE (approx. 6-7 min) and saves:
  results/nasbench101/audit/gt_accuracies.npy  -- shape (423624,) float32, values 0-100
  results/nasbench101/audit/param_counts.npy   -- shape (423624,) int64
  results/nasbench101/audit/arch_hashes.npy    -- shape (423624,) object (Python str)
  results/nasbench101/audit/arch_specs.pkl     -- dict hash -> {'adjacency': np.int8 (7,7),
                                                                  'ops': list[str]}
  results/nasbench101/audit/audit_summary.json -- stats and sanity checks

GT accuracy = mean of final_test_accuracy across 3 seeds at 108 epochs, multiplied by 100.
"""

import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import json
import pickle
import time
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT_DIR    = Path("/home/anan/NAS/Experimentation/Data-Agnostic-NAS-Experimentation")
DATA_FILE   = ROOT_DIR / "data/nas101/nasbench_full.tfrecord"
OUT_DIR     = ROOT_DIR / "results/nasbench101/audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_EVERY   = 10000   # print progress every N architectures
EPOCHS      = 108     # only 108-epoch results exist for all archs


def main():
    # -----------------------------------------------------------------------
    # 1. Load nasbench API
    # -----------------------------------------------------------------------
    print("Loading nasbench API ...", flush=True)
    t0 = time.time()

    from nasbench import api  # patched: see scripts/nasbench101/README.md
    nb = api.NASBench(str(DATA_FILE))

    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s", flush=True)

    # -----------------------------------------------------------------------
    # 2. Iterate all architectures
    # -----------------------------------------------------------------------
    hashes_list   = list(nb.hash_iterator())
    n_total       = len(hashes_list)
    print(f"  Total architectures: {n_total}", flush=True)

    gt_accuracies = np.zeros(n_total, dtype=np.float32)
    param_counts  = np.zeros(n_total, dtype=np.int64)
    arch_hashes   = np.empty(n_total, dtype=object)
    arch_specs    = {}   # hash -> {'adjacency': np.int8, 'ops': list}

    n_missing_seeds   = 0   # archs with fewer than 3 seeds
    n_degenerate      = 0   # archs with GT < 20% (likely degenerate / disconnected)

    print(f"\nExtracting data from {n_total} architectures ...", flush=True)
    t_iter = time.time()

    for idx, h in enumerate(hashes_list):
        if idx % LOG_EVERY == 0 and idx > 0:
            elapsed = time.time() - t_iter
            rate    = idx / elapsed
            eta     = (n_total - idx) / rate
            print(f"  [{idx}/{n_total}] elapsed={elapsed:.0f}s  ETA={eta:.0f}s", flush=True)

        fixed, computed = nb.get_metrics_from_hash(h)

        # -- GT accuracy (mean over seeds at 108 epochs) --------------------
        seed_records = computed[EPOCHS]           # list of 3 dicts
        seed_accs = [rec["final_test_accuracy"] for rec in seed_records]
        gt_frac   = float(np.mean(seed_accs))    # 0.0 - 1.0

        if len(seed_accs) < 3:
            n_missing_seeds += 1

        gt_pct = gt_frac * 100.0

        # -- Store ----------------------------------------------------------
        gt_accuracies[idx] = gt_pct
        param_counts[idx]  = int(fixed["trainable_parameters"])
        arch_hashes[idx]   = h
        arch_specs[h] = {
            "adjacency": fixed["module_adjacency"].astype(np.int8),
            "ops":       list(fixed["module_operations"]),
        }

        if gt_pct < 20.0:
            n_degenerate += 1

    total_time = time.time() - t_iter
    print(f"\nIteration complete in {total_time:.1f}s ({total_time/60:.1f} min)", flush=True)

    # -----------------------------------------------------------------------
    # 3. Sanity check: linear regression of log(params) -> GT
    # -----------------------------------------------------------------------
    from scipy.stats import spearmanr
    log_params = np.log(param_counts.astype(np.float64))

    # R-squared via numpy polyfit
    coeffs  = np.polyfit(log_params, gt_accuracies, 1)
    pred    = np.polyval(coeffs, log_params)
    ss_res  = np.sum((gt_accuracies - pred) ** 2)
    ss_tot  = np.sum((gt_accuracies - gt_accuracies.mean()) ** 2)
    r2      = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    rho, pval = spearmanr(log_params, gt_accuracies)

    # Top-1% std
    top1_n      = max(1, int(0.01 * n_total))
    top1_thresh = np.sort(gt_accuracies)[-top1_n]
    top1_accs   = gt_accuracies[gt_accuracies >= top1_thresh]
    top1_std    = float(np.std(top1_accs))

    print(f"\n=== Audit Summary ===", flush=True)
    print(f"  n_total:          {n_total}", flush=True)
    print(f"  n_missing_seeds:  {n_missing_seeds}", flush=True)
    print(f"  n_degenerate:     {n_degenerate}  (GT < 20%)", flush=True)
    print(f"  GT mean:          {gt_accuracies.mean():.4f}%", flush=True)
    print(f"  GT std:           {gt_accuracies.std():.4f}%", flush=True)
    print(f"  GT min:           {gt_accuracies.min():.4f}%", flush=True)
    print(f"  GT max:           {gt_accuracies.max():.4f}%", flush=True)
    print(f"  R2 (log_params->GT): {r2:.4f}  (expect ~0.047)", flush=True)
    print(f"  Spearman rho:     {rho:.4f}  (expect ~0.435)", flush=True)
    print(f"  Top-1% std:       {top1_std:.4f}%", flush=True)

    # -----------------------------------------------------------------------
    # 4. Save outputs
    # -----------------------------------------------------------------------
    print("\nSaving outputs ...", flush=True)

    np.save(OUT_DIR / "gt_accuracies.npy",  gt_accuracies)
    print(f"  Saved gt_accuracies.npy  shape={gt_accuracies.shape}  dtype={gt_accuracies.dtype}", flush=True)

    np.save(OUT_DIR / "param_counts.npy",   param_counts)
    print(f"  Saved param_counts.npy   shape={param_counts.shape}   dtype={param_counts.dtype}", flush=True)

    np.save(OUT_DIR / "arch_hashes.npy",    arch_hashes,  allow_pickle=True)
    print(f"  Saved arch_hashes.npy    shape={arch_hashes.shape}    dtype={arch_hashes.dtype}", flush=True)

    with open(OUT_DIR / "arch_specs.pkl", "wb") as f:
        pickle.dump(arch_specs, f, protocol=4)
    print(f"  Saved arch_specs.pkl     keys={len(arch_specs)}", flush=True)

    # Also save a plain-text hash list for easy inspection
    with open(OUT_DIR / "arch_hashes.txt", "w") as f:
        for h in arch_hashes:
            f.write(h + "\n")
    print(f"  Saved arch_hashes.txt    lines={n_total}", flush=True)

    # Summary JSON
    summary = {
        "n_total":             int(n_total),
        "n_missing_seeds":     int(n_missing_seeds),
        "n_degenerate_lt20":   int(n_degenerate),
        "gt_mean_pct":         float(gt_accuracies.mean()),
        "gt_std_pct":          float(gt_accuracies.std()),
        "gt_min_pct":          float(gt_accuracies.min()),
        "gt_max_pct":          float(gt_accuracies.max()),
        "r2_logparams_vs_gt":  r2,
        "spearman_rho":        float(rho),
        "spearman_pval":       float(pval),
        "top1pct_std_pct":     top1_std,
        "top1pct_n":           int(len(top1_accs)),
        "load_time_s":         round(load_time, 1),
        "iteration_time_s":    round(total_time, 1),
    }

    with open(OUT_DIR / "audit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved audit_summary.json", flush=True)

    print(f"\nStep 0 complete.  Outputs in: {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
