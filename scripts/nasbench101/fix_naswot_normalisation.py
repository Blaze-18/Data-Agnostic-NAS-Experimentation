"""
Fix NASWOT and ZenScore scores by replacing the variable denominator
(number of Conv2d layers actually hooked per architecture) with a fixed
denominator (maximum possible Conv2d layers in any NAS-Bench-101 architecture).

ROOT CAUSE:
  Current: score = mean(cov_trace) = sum(traces) / n_conv_hooked_per_arch
  Problem: n_conv_hooked varies by architecture (maxpool ops have no Conv2d).
           Architectures with more maxpool -> fewer hooked layers -> higher mean
           -> negative correlation with GT (maxpool-heavy arches are worse).
  Fix:     score = sum(traces) / MAX_CONV_LAYERS
           where MAX_CONV_LAYERS = fixed constant (same for all archs).

MAX_CONV_LAYERS derivation (NASBench101Net, C_BASE=16):
  - Stem:          1  Conv2d (always present)
  - Downsample x2: 2  Conv2d (always present: Downsample.proj)
  - Per cell:      up to 5 intermediate nodes (7-node DAG: nodes 1-5)
                   each can be conv3x3 (1 Conv2d) or conv1x1 (1 Conv2d)
  - 9 cells total  (3 stacks × 3 cells/stack)
  - Max cell Conv2d: 5 × 9 = 45
  - TOTAL MAX:     1 + 2 + 45 = 48

Correction formula (derived without GPU):
  corrected = raw_mean * n_conv_hooked_per_arch / 48

n_conv_hooked_per_arch is computed from arch_specs.pkl:
  count = 1 (stem) + 2 (downsamples)
        + 9 * (number of intermediate nodes in the cell DAG that are conv AND
               have at least one predecessor in the adjacency matrix)

This script:
  1. Computes n_conv_hooked for all 423,624 architectures from arch_specs.pkl
  2. Loads raw naswot.npy and zenscore.npy (computed with mean formula)
  3. Applies the correction: score = raw_mean * n_conv / 48
  4. Overwrites naswot.npy and zenscore.npy with corrected values
  5. Then Steps 2-6 can be re-run on corrected scores

Runtime: ~60 seconds (CPU only, no GPU needed).
"""

import numpy as np
import pickle
from pathlib import Path
from scipy.stats import spearmanr

ROOT_DIR    = Path("/home/anan/NAS/Experimentation/Data-Agnostic-NAS-Experimentation")
AUDIT_DIR   = ROOT_DIR / "results/nasbench101/audit"
RAW_DIR     = ROOT_DIR / "results/nasbench101/raw_proxy_scores"

CONV_OPS     = {"conv3x3-bn-relu", "conv1x1-bn-relu"}
MAX_CONV     = 48   # stem(1) + downsamples(2) + 9 cells × 5 nodes(45)
N_CELLS      = 9    # 3 stacks × 3 cells/stack


def count_conv_layers(adjacency: np.ndarray, ops: list) -> int:
    """
    Count the number of Conv2d layers that would be hooked in the proxy model
    for a given NAS-Bench-101 cell spec.

    Fixed layers (always present):
      - 1 stem Conv2d
      - 2 downsample Conv2d (Downsample.proj × 2)

    Variable layers (per cell, replicated 9 times):
      For each intermediate node j (1 <= j <= N-2):
        - Include if ops[j] is a conv op AND node j has at least one predecessor
          (i.e., any(adjacency[:j, j]) == True)

    Args:
        adjacency: (N, N) int8 array, upper triangular
        ops:       list of N operation strings

    Returns:
        Total Conv2d count for this architecture
    """
    N = adjacency.shape[0]
    conv_per_cell = 0
    for j in range(1, N - 1):          # intermediate nodes only
        has_pred = bool(adjacency[:j, j].any())
        if has_pred and ops[j] in CONV_OPS:
            conv_per_cell += 1

    return 1 + 2 + N_CELLS * conv_per_cell


def main():
    print("Loading arch_specs ...", flush=True)
    with open(AUDIT_DIR / "arch_specs.pkl", "rb") as f:
        arch_specs = pickle.load(f)

    arch_hashes = np.load(AUDIT_DIR / "arch_hashes.npy", allow_pickle=True)
    n_total     = len(arch_hashes)
    gt          = np.load(AUDIT_DIR / "gt_accuracies.npy").astype(np.float64)

    print(f"  {n_total} architectures\n", flush=True)

    # ── Step 1: compute n_conv for every arch ────────────────────────────────
    print("Computing n_conv_hooked per architecture ...", flush=True)
    n_conv = np.zeros(n_total, dtype=np.int32)

    for i, h in enumerate(arch_hashes):
        spec       = arch_specs[str(h)]
        n_conv[i]  = count_conv_layers(spec["adjacency"], spec["ops"])
        if (i + 1) % 50000 == 0:
            print(f"  {i+1}/{n_total}", flush=True)

    print(f"\nn_conv stats:")
    print(f"  min={n_conv.min()}  max={n_conv.max()}  mean={n_conv.mean():.2f}  "
          f"median={np.median(n_conv):.0f}")

    # Confirm max ≤ MAX_CONV
    assert n_conv.max() <= MAX_CONV, \
        f"n_conv.max()={n_conv.max()} exceeds MAX_CONV={MAX_CONV} — update MAX_CONV!"
    print(f"  MAX_CONV={MAX_CONV}  (verified: n_conv.max() <= MAX_CONV ✓)\n")

    # ── Step 2: load and inspect raw scores ──────────────────────────────────
    raw_naswot   = np.load(RAW_DIR / "naswot.npy").astype(np.float64)
    raw_zenscore = np.load(RAW_DIR / "zenscore.npy").astype(np.float64)

    rho_naswot_before,  _ = spearmanr(raw_naswot,   gt)
    rho_zenscore_before,_ = spearmanr(raw_zenscore, gt)
    print(f"BEFORE correction:")
    print(f"  NASWOT   ρ with GT = {rho_naswot_before:.4f}")
    print(f"  ZenScore ρ with GT = {rho_zenscore_before:.4f}")

    # ── Step 3: apply correction ──────────────────────────────────────────────
    # current   = sum_traces / n_conv
    # desired   = sum_traces / MAX_CONV
    # corrected = current * n_conv / MAX_CONV
    scale       = n_conv.astype(np.float64) / MAX_CONV
    naswot_fixed   = raw_naswot   * scale
    zenscore_fixed = raw_zenscore * scale

    rho_naswot_after,  _ = spearmanr(naswot_fixed,   gt)
    rho_zenscore_after,_ = spearmanr(zenscore_fixed, gt)

    print(f"\nAFTER correction (÷ MAX_CONV={MAX_CONV}):")
    print(f"  NASWOT   ρ with GT = {rho_naswot_after:.4f}")
    print(f"  ZenScore ρ with GT = {rho_zenscore_after:.4f}")

    print(f"\nSign change: NASWOT  {rho_naswot_before:.4f} → {rho_naswot_after:.4f}  "
          f"({'✅ fixed' if rho_naswot_after > 0 else '❌ still negative'})")
    print(f"Sign change: ZenScore {rho_zenscore_before:.4f} → {rho_zenscore_after:.4f}  "
          f"({'✅ fixed' if rho_zenscore_after > 0 else '❌ still negative'})")

    # ── Step 4: sanity checks ─────────────────────────────────────────────────
    assert naswot_fixed.shape   == (n_total,), "Shape mismatch naswot"
    assert zenscore_fixed.shape == (n_total,), "Shape mismatch zenscore"
    assert np.isfinite(naswot_fixed).all(),    "Non-finite values in naswot_fixed"
    assert np.isfinite(zenscore_fixed).all(),  "Non-finite values in zenscore_fixed"
    assert (naswot_fixed >= 0).all(),          "Negative values in naswot_fixed"
    print("\n✅ All sanity checks passed")

    # ── Step 5: overwrite raw score files ─────────────────────────────────────
    # Back up originals first
    orig_naswot_path   = RAW_DIR / "naswot_original_mean.npy"
    orig_zen_path      = RAW_DIR / "zenscore_original_mean.npy"
    if not orig_naswot_path.exists():
        np.save(orig_naswot_path,   raw_naswot)
        print(f"Backed up original NASWOT   → {orig_naswot_path.name}")
    if not orig_zen_path.exists():
        np.save(orig_zen_path,      raw_zenscore)
        print(f"Backed up original ZenScore → {orig_zen_path.name}")

    np.save(RAW_DIR / "naswot.npy",    naswot_fixed.astype(np.float64))
    np.save(RAW_DIR / "zenscore.npy",  zenscore_fixed.astype(np.float64))
    np.save(RAW_DIR / "n_conv_per_arch.npy", n_conv)
    print(f"\nSaved corrected naswot.npy and zenscore.npy to {RAW_DIR}")
    print(f"Saved n_conv_per_arch.npy  (shape={n_conv.shape}, for reference)")

    print(f"\n{'='*60}")
    print("NEXT STEPS: re-run Steps 2-6 to propagate the corrected scores")
    print("  python scripts/nasbench101/transform_proxies_101.py")
    print("  python scripts/nasbench101/analyze_distributions_101.py")
    print("  python scripts/nasbench101/validate_ranking_correlations_101.py")
    print("  python scripts/nasbench101/bias_disentanglement_101.py")
    print("  python scripts/nasbench101/pca_whitening_101.py")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
