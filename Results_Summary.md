# Results Summary — NAS-Bench-101

---

## Step 7: Surrogate MLP — Ablation Results

**Date:** 2026-05-09
**Script:** `scripts/nasbench101/train_mlp_101.py`
**Outputs:** `results/nasbench101/surrogate_mlp/`

### Setup

| Parameter | Value |
|---|---|
| Architecture | k → 64 (ReLU) → 32 (ReLU) → 1 |
| Loss | RankNet log-sigmoid pairwise ranking loss |
| Optimiser | Adam, lr = 1e-3 |
| Split | 80% train / 10% val / 10% test (seed = 42) |
| Early stopping | patience = 20 epochs, max 500 epochs |
| Dataset size | 423,624 architectures |
| Device | CUDA (RTX 5070 Ti) |

---

### Ablation Table

| Variant | Features | Early stop (epoch) | ρ global | ρ competitive (GT > 50%) | Top-5% prec. | Top-10% prec. |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `size_only` — log(param_count) only | 1 | 59 | 0.5559 | 0.5580 | 0.201 | 0.279 |
| `best_raw` — SynFlow only (best activation proxy) | 1 | 30 | 0.4298 | 0.4283 | 0.142 | 0.210 |
| `pca_raw` — PCA on all 4 raw proxies, no debiasing | 3 | 76 | 0.6275 | 0.6287 | 0.388 | 0.452 |
| **`full_pipeline`** — debiased + whitened (proposed) | **3** | **95** | **0.6494** | **0.6505** | 0.375 | 0.451 |

*ρ = Spearman rank correlation. Competitive subset = architectures with GT accuracy > 50%.*

---

### Key Findings

**1. Single activation proxies are weaker than raw size.**
SynFlow alone (ρ = 0.430) is worse than log(param_count) alone (ρ = 0.556). Activation-based proxies in isolation carry less ranking signal than the structural size prior on NAS-Bench-101, due to their high collinearity with param_count (SynFlow–param_count Spearman ρ = 0.543, confirmed in Step 5).

**2. Multi-proxy fusion gives a large gain (+7.2pp).**
Combining all four raw proxies via PCA (`pca_raw`, ρ = 0.628) substantially outperforms the best single proxy (`size_only`, ρ = 0.556). The proxies carry complementary information that no single proxy captures alone.

**3. Debiasing adds genuine signal above fusion alone (+2.2pp).**
`full_pipeline` (ρ = 0.649) beats `pca_raw` (ρ = 0.628) despite using the same number of features (3) and the same MLP. The only difference is the bias-disentanglement step applied before PCA whitening. This gap is the direct measurable contribution of the proposed method.

**4. End-to-end improvement over the size baseline: +9.3pp.**
The full pipeline lifts Spearman ρ from 0.556 (param_count alone) to 0.649.

**5. Top-k precision nuance.**
`full_pipeline` top-5% precision (0.375) is marginally below `pca_raw` (0.388). This is expected: the debiasing step deliberately suppresses the size-collinear component. The very top architectures on NAS-Bench-101 tend to be large, so removing the size bias slightly reshuffles the extreme tail. The proposed method is globally better calibrated (higher ρ over all 423k architectures) at the cost of a small redistribution at the top-5% tail.

---

### Marginal Gains Summary

| Comparison | Δρ (global) | Interpretation |
|---|:-:|---|
| `size_only` → `best_raw` | −0.126 | Activation proxy alone is weaker than size |
| `size_only` → `pca_raw` | +0.072 | Multi-proxy fusion value |
| `pca_raw` → `full_pipeline` | +0.022 | **Debiasing contribution (core claim)** |
| `size_only` → `full_pipeline` | +0.093 | Full system improvement |

---

### Output Files

| File | Description |
|---|---|
| `mlp_results.json` | Full metrics + loss history for all 4 variants |
| `ablation_table.json` | Compact ablation table (for reporting) |
| `training_curves.png` | Train/val loss per epoch for each variant |
| `ablation_comparison.png` | Grouped bar chart: ρ_global, ρ_competitive, top-5%, top-10% |
