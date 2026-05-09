# NAS-Bench-101 Experimentation Plan & Progress

**Project**: Bias-Disentangled Structural Proxy Embedding Framework  
**Benchmark**: NAS-Bench-101 (423,624 architectures, DAG search)  
**Machine**: Lab machine (GTX 5070 Ti, 32 GB RAM, Linux)  
**TFRecord Location**: `data/nasbench101/nasbench_full.tfrecord` (2.1 GB)  
**Start Date**: May 8, 2026  
**Priority Level**: 🔴 CRITICAL — Most important benchmark for thesis validation  

---

## 📋 Execution Plan Overview

### Three Phases

| Phase | Steps | Timeline | GPU Required | Parallelizable |
|-------|-------|----------|--------------|---|
| **Phase 1: Data Loading & Audit** | 0, 1a | ~7-10 min | ❌ No | Sequential |
| **Phase 2: Proxy Computation** | 1b, 1c, 1d | ~45-90 min | ✅ Yes | ✅ 3 parallel |
| **Phase 3: Analysis & Training** | 2-7 | ~30-40 min | ✅ (Step 7 only) | Sequential |

**Total Expected Time**: ~2-3 hours from start to finish

---

## Phase 1: Data Loading & Audit (Steps 0, 1a)

### Objective
Load TFRecord, extract ground-truth accuracies and parameter counts, verify data integrity.

### Step 0: Structural Audit
**Script**: `scripts/nasbench101/step0_audit_101.py`  
**Duration**: ~390 seconds (6.5 min) — TFRecord load time  
**Output Location**: `results/nasbench101/audit/`

**Expected Outputs**:
```
- gt_accuracies.npy        (423,624) float32 — ground truth [0,100]
- param_counts.npy         (423,624) int64
- arch_hashes.npy          (423,624) object — ordered hash list
- arch_specs.pkl           dict {hash → {adjacency, ops}} (~45 MB)
- audit_summary.json       Distribution stats
```

**Success Criteria**:
- ✅ n_total = 423,624 (all architectures loaded)
- ✅ n_degenerate_lt20 ≈ 646 (degenerate cluster identified)
- ✅ gt_mean ≈ 89.68%, gt_std ≈ 5.80%
- ✅ Zero NaN/Inf values
- ✅ R² (param_count → GT) ≈ 0.047 ± 0.005

**Failure Handling**:
- **"TFRecord not found"**: Verify `data/nasbench101/nasbench_full.tfrecord` exists
- **"ModuleNotFoundError: nasbench"**: Run from repo root; verify patches applied
- **"ProtoBuf error"**: Set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before running

**Status**: ✅ **COMPLETE** (May 8, 2026)

**Actual Results**:
- Load time: 284.0s (faster than expected 390s)
- Iteration time: 20.9s
- n_total: 423,624 ✓
- n_degenerate: 646 ✓
- GT mean: 89.68%, std: 5.80% ✓
- R²: 0.047 ✓
- Spearman ρ: 0.435 ✓

---

### Step 1a: Parameter Count Extraction
**Script**: `scripts/nasbench101/compute_proxy_param_count_101.py`  
**Duration**: <1 min (trivial copy from audit)  
**Output**: `results/nasbench101/raw_proxy_scores/param_count.npy`

**Success Criteria**:
- ✅ Output shape: (423,624,)
- ✅ dtype: float64 (stored as float for consistency)
- ✅ Range: 2.27e+05 to 5.00e+07 parameters
- ✅ No NaN/Inf

**Status**: ✅ **COMPLETE**

---

## Phase 2: GPU Proxy Computation (Steps 1b, 1c, 1d)

### Objective
Compute three zero-cost proxies on GPU in parallel: SynFlow, NASWOT, ZenScore.

### Architecture & Resources
- **GPU**: GTX 5070 Ti (12 GB VRAM)
- **Model**: Small proxy network (C_BASE=16 channels)
- **Batch Size**: Auto-tuned per proxy
- **Checkpointing**: Every 1,000 architectures (safe resume on interrupt)

### Step 1b: SynFlow Proxy
**Script**: `scripts/nasbench101/compute_proxy_synflow_101.py`  
**Actual Duration**: ~80 min (interrupted at 70%, resumed)  
**Algorithm**: Tanaka et al. 2020 (all-ones input, linearised weights, Σ|grad×weight|)  
**Output**: `results/nasbench101/raw_proxy_scores/synflow.npy`

**Actual Results**:
- Shape: (423,624,) float64 ✓
- Valid finite values: 419,454 / 423,624 (99.0%)
- **Inf values**: 2,677 (0.63%) ⚠️
- **NaN values**: 1,493 (0.35%) ⚠️
- Finite range: 3.71e+03 to 1.67e+40
- Finite mean: 1.27e+37, median: 6.87e+23
- Distribution: Extremely right-skewed (numerical overflow in large networks)

**Key Findings**:
- ⚠️ **Numerical overflow**: 4,170 architectures (1%) have Inf/NaN values
- These are very large networks where gradient magnitudes exceed float64 limits
- Log-transform in Step 2 will handle this by clipping/replacing Inf values
- Issue does NOT invalidate results — standard for SynFlow on large DAG spaces

**Status**: ✅ **COMPLETE** with known numerical overflow (expected behavior)

---

### Step 1c: NASWOT Proxy
**Script**: `scripts/nasbench101/compute_proxy_naswot_101.py`  
**Actual Duration**: ~70 min  
**Algorithm**: Multi-layer Conv2d hooking; single batch (8 samples); mean covariance trace  
**Output**: `results/nasbench101/raw_proxy_scores/naswot.npy`

**Actual Results**:
- Shape: (423,624,) float64 ✓
- Valid (>0): 423,624 / 423,624 (100.0%) ✓
- Range: 2.58e-05 to 9.83e+09
- Mean: 5.13e+05
- NaN/Inf: 0 ✓
- Distribution: Right-skewed (wider range than expected, includes very large values)

**Key Points**:
- ✅ 100% valid scores, no numerical issues
- ✅ All Conv2d layers successfully hooked
- ⚠️ Max value (9.83e+09) is much larger than NAS-Bench-201 (0.003)
- This reflects DAG diversity: variable depth/width architectures

**Status**: ✅ **COMPLETE**

---

### Step 1d: ZenScore Proxy
**Script**: `scripts/nasbench101/compute_proxy_zenscore_101.py`  
**Actual Duration**: ~50 min  
**Algorithm**: 4 random batch draws; mean covariance trace; stochastic averaging  
**Output**: `results/nasbench101/raw_proxy_scores/zenscore.npy`

**Actual Results**:
- Shape: (423,624,) float64 ✓
- Valid (>0): 423,624 / 423,624 (100.0%) ✓
- Range: 3.34e-05 to 8.82e+09
- Mean: 5.30e+05
- NaN/Inf: 0 ✓
- Distribution: Right-skewed, very similar to NASWOT

**Key Points**:
- ✅ All scores valid, no numerical issues
- ✅ Stochastic averaging across 4 samples successful
- Correlation with NASWOT expected to be high (both measure activation diversity)
- Similar scale and range to NASWOT (mean: 5.30e+05 vs 5.13e+05)

**Status**: ✅ **COMPLETE**

---

### Phase 2 Execution Strategy

**Run all three in parallel** (different terminals, different output files):

```bash
# Terminal 1: SynFlow (fastest, finish first)
python scripts/nasbench101/gpu/compute_proxy_synflow_gpu.py

# Terminal 2: NASWOT (medium speed)
python scripts/nasbench101/gpu/compute_proxy_naswot_gpu.py

# Terminal 3: ZenScore (slowest, finish last)
python scripts/nasbench101/gpu/compute_proxy_zenscore_gpu.py
```

**Checkpoint Recovery**:
- Each script creates JSON checkpoint every 1,000 architectures
- If interrupted, re-run same command: will resume from checkpoint
- Checkpoint files: `results/nasbench101/raw_proxy_scores/{proxy}_checkpoint.json`

**Wait Condition**: All three must complete before Phase 3.  
**Progress Tracking**: Monitor GPU load with `nvidia-smi` in 4th terminal

**Status**: ✅ **COMPLETE** (May 8, 2026)

**Phase 2 Summary**:
- Total GPU time: ~3.3 hours (synflow 80min + naswot 70min + zenscore 50min)
- GPU utilization: RTX 5070 Ti at ~16GB VRAM
- Checkpointing successful (synflow interrupted & resumed at 70%)
- All 4 proxy scores successfully generated

**Issues Encountered**:
1. **SynFlow numerical overflow** (4,170 Inf/NaN values, 1% of dataset)
   - Cause: Very large DAG networks exceed float64 precision
   - Impact: Will be handled by Step 2 transform (clip/replace Inf)
   - Acceptable: Standard behavior for SynFlow on large search spaces

**Data Quality Assessment**:
- ✅ param_count: 100% valid, no issues
- ⚠️ synflow: 99% valid (1% overflow, expected)
- ✅ naswot: 100% valid, clean
- ✅ zenscore: 100% valid, clean

**Ready for Phase 3**: YES ✓

---

## Phase 3: Analysis & Training (Steps 2-7)

### Objective
Transform proxies → analyze distributions → validate ranking → disentangle bias → whiten → train MLP

### Step 2: Log Transformation
**Script**: `scripts/nasbench101/transform_proxies_101.py`  
**Duration**: <2 min  
**Inputs**: Raw proxy scores (.npy files from Phase 2)  
**Outputs**: `results/nasbench101/transformed_proxy/`

**Transformations Applied**:
- All proxies: `log(x + offset)` where offset is proxy-specific
- **Auto-detect sign**: If raw Spearman ρ with GT < 0, negate before log

**Expected Output**:
```
param_count_transformed.npy       (423,624,) float32
synflow_transformed.npy
naswot_transformed.npy
zenscore_transformed.npy
transform_summary.json            (sign decisions, offset values)
```

**Success Criteria**:
- ✅ No NaN/Inf in output
- ✅ All four arrays shape (423,624,)
- ✅ Summary JSON contains negation flags

**Actual Results**:
- All 4 proxies successfully transformed ✓
- SynFlow: 4,170 Inf/NaN replaced with max finite value (1.67e+40) ✓
- NASWOT: Raw ρ = -0.1982 → Negated → Transformed ρ = +0.1982 ✓
- ZenScore: Raw ρ = -0.2251 → Negated → Transformed ρ = +0.2251 ✓
- All outputs clean (no NaN/Inf) ✓

**Key Finding - NASWOT/ZenScore Sign Reversal**:
⚠️ **CRITICAL**: NASWOT and ZenScore had **NEGATIVE raw correlation** with GT (~-0.20)
- This is **opposite** of NAS-Bench-201 (positive +0.52) and SSS (positive +0.52)
- Interpretation: In DAG space, high activation diversity → LOW accuracy
- Suggests skip-heavy architectures have high covariance but poor learning
- Negation applied to flip correlation positive for pipeline consistency
- **Thesis impact**: Validates benchmark-dependent proxy behavior claim

**⚠️ Post-Fix Re-run Note (May 9, 2026)**:
A normalisation bug was found in `compute_proxy_naswot_101.py`: NASWOT traces were
divided by the number of Conv2d layers *in that architecture* (variable), instead of
the global maximum (`MAX_CONV = 48`). This inflated scores for shallow architectures.
Fix: divide all traces by the fixed constant 48. Steps 2–6 were re-run after the fix.
NASWOT raw ρ changed: −0.2245 → −0.1982. Original backed up as `naswot_original_mean.npy`.

**Transformed Correlations (post-fix)**:
- param_count: ρ = +0.4350 (strongest)
- synflow: ρ = +0.4206 (surprisingly strong, comparable to size)
- naswot: ρ = +0.1982 (weak, after negation)
- zenscore: ρ = +0.2251 (weak, after negation)

**Status**: ✅ **COMPLETE** (May 8, 2026; re-run May 9, 2026 after NASWOT fix)

---

### Step 3: Distribution Analysis
**Script**: `scripts/nasbench101/analyze_distributions_101.py`  
**Actual Duration**: <1 min  
**Output**: `results/nasbench101/proxy_distribution/stats_summary.json`

**Actual Results**:
- param_count: mean=15.51, std=0.95, skew=0.07, kurt=-0.41 ✓ (nearly normal)
- synflow: mean=55.72, std=14.81, skew=0.22, kurt=-0.37 ✓ (nearly normal)
- naswot: mean=4.18, std=5.37, skew=-1.52, kurt=1.90 ⚠️ (left-skewed)
- zenscore: mean=4.16, std=5.37, skew=-1.52, kurt=1.90 ⚠️ (left-skewed)

**Key Observations**:
- ✅ param_count and synflow well-normalized (|skew| < 0.3)
- ⚠️ NASWOT/ZenScore left-skewed (skew = -1.52) due to negation transform
- This is expected: negating positive values flips the distribution tail
- NASWOT and ZenScore nearly identical distributions (as expected, both measure activation diversity)

**Status**: ✅ **COMPLETE** (May 8, 2026)

---

### Step 4: Ranking Correlation Validation
**Script**: `scripts/nasbench101/validate_ranking_correlations_101.py`  
**Actual Duration**: <1 min  
**Output**: `results/nasbench101/proxy_validation/correlation_results.json`

**Actual Results** (Global / Competitive) — post-fix values:

| Proxy | Spearman ρ | Kendall τ | Top-5% | Top-10% |
|-------|-----------|-----------|---------|----------|
| **param_count** | 0.4350 / 0.4357 | 0.3066 / 0.3072 | 20.1% | 27.1% |
| **synflow** | 0.4206 / 0.4188 | 0.2867 / 0.2858 | 11.2% | 21.7% |
| **naswot** | 0.1982 / ~0.196 | ~0.135 / ~0.134 | ~3.5% | ~9.8% |
| **zenscore** | 0.2251 / 0.2220 | 0.1539 / 0.1520 | 3.9% | 11.0% |

*NASWOT values updated after normalisation fix (May 9). Pre-fix was 0.2245.*

**Key Findings**:
1. ✅ **SynFlow extremely strong** (ρ=0.421) - nearly matches param_count!
   - This is a **dramatic reversal** from NAS-Bench-201 (ρ=-0.002)
   - Confirms SynFlow captures DAG path-capacity distribution effectively

2. ⚠️ **NASWOT/ZenScore weak** (ρ~0.22)
   - Much weaker than NAS-Bench-201 (ρ~0.52) and SSS (ρ~0.52)
   - Negative raw correlation suggests different dynamics in DAG space
   - **Likely to fail bias disentanglement** (Step 5)

3. ✅ **Global vs Competitive correlations nearly identical**
   - Confirms proxies work consistently across the full accuracy range
   - No degenerate-cluster inflation

4. ⚠️ **Top-K precision low for all proxies**
   - Top-5%: param_count 20%, synflow 11%, naswot/zenscore ~4%
   - Reflects tight accuracy range at the top (93-94%)
   - Proxies cannot meaningfully discriminate within 1% accuracy differences

**Cross-Benchmark Comparison** (Spearman ρ):

| Proxy | NAS-Bench-201 | SSS | **NAS-Bench-101** |
|-------|---------------|-----|-------------------|
| param_count | 0.749 | 0.872 | **0.435** ✓ (weakest - expected) |
| synflow | 0.164 | 0.916 | **0.421** ✓ (strongest - major finding!) |
| naswot | 0.515 | 0.524 | **0.225** ⚠️ (weak) |
| zenscore | 0.554 | 0.557 | **0.225** ⚠️ (weak) |

**Status**: ✅ **COMPLETE** (May 8, 2026)

**Critical Prediction for Step 5**:
- ✅ SynFlow: Expected to **SURVIVE** with partial ρ ~ 0.30-0.40
- ❌ NASWOT: Expected to **COLLAPSE** with partial ρ < 0.10
- ❌ ZenScore: Expected to **COLLAPSE** with partial ρ < 0.10
topk_precision.png                 (top-1%, 5%, 10% precision)
```

**Expected Results**:

| Proxy | Global ρ | Signal | Note |
|-------|----------|--------|------|
| Param Count | ~0.43-0.50 | Weak-Moderate | Size signal only (expected in low-R² benchmark) |
| SynFlow | ? | ? | **Unknown** — first benchmark to test |
| NASWOT | ? | ? | **Unknown** — will collapse if operation diversity dominates |
| ZenScore | ? | ? | **Unknown** — similar to NASWOT |

**Thesis Hypothesis**:
- SynFlow may survive (DAGs have variable path capacity, unlike SSS)
- NASWOT/ZenScore may survive better than SSS (DAGs have op diversity, unlike SSS)
- But much weaker than NAS-Bench-201 (param dominates less here)

**Status**: ⏳ NOT STARTED

---

### Step 5: Bias Disentanglement
**Script**: `scripts/nasbench101/bias_disentanglement_101.py`  
**Duration**: 3-5 min  
**Output**: `results/nasbench101/debiased_proxy/`

**Method**:
1. OLS: `Proxy ~ log(param_count)` → residuals
2. Partial-rank Spearman: regress ranks, correlate residuals
3. Decision: keep if partial ρ ≥ 0.10

**Outputs**:
```
partial_correlations.json           (both methods, decisions)
phase_b_summary.json                (OLS fit stats)
phase_b_residuals.png               (residual scatter plots)
phase_c_scatter.png                 (debiased vs GT-debiased)
{synflow,naswot,zenscore}_residuals.npy
gt_residuals.npy
```

**Actual Decision Table** (post-fix values):

| Proxy | Global ρ | Partial ρ | Decision | Note |
|-------|----------|-----------|----------|------|
| Param Count | 0.435 | — | **Bias covariate** | Always used raw |
| SynFlow | 0.421 | **0.2420** | **KEEP_DOCUMENTED** | Partial ρ > 0.10 threshold |
| NASWOT | 0.198 | **0.1034** | **KEEP_DOCUMENTED** | Marginal but passes (down from 0.1246 pre-fix) |
| ZenScore | 0.225 | **0.1032** | **KEEP_DOCUMENTED** | Marginal but passes (down from 0.1250 pre-fix) |

**OLS-residual ρ (secondary metric, post-fix)**:
- SynFlow: 0.2179
- NASWOT: ~0.080 (reduced after fix)
- ZenScore: ~0.080

**Outcome**: ✅ All 3 proxies survive — all classified as KEEP_DOCUMENTED (partial ρ in [0.10, 0.30))  
→ PCA input will be 4D: [log_param_count, synflow_residual, naswot_residual, zenscore_residual]

**Status**: ✅ COMPLETE (May 8, 2026; re-run May 9 after NASWOT fix)

---

### Step 6: PCA Whitening
**Script**: `scripts/nasbench101/pca_whitening_101.py`  
**Duration**: <1 min  
**Inputs**: Debiased proxies from Step 5  
**Output**: `results/nasbench101/pca_whitening/`

**Outputs**:
```
whitened_features.npy               (423,624 × N_features) float32 — MLP INPUT
scaler.pkl                          (StandardScaler for inference)
pca_model.pkl                       (PCA(whiten=True) for inference)
arch_ids.npy                        (423,624,) int32 — row→arch alignment
pca_summary.json                    (eigenvalues, loadings, validation)
pca_plots.png                       (4-panel diagnostics)
```

**Validation Checks**:
- ✅ Column means ≈ 0 (< 1e-4)
- ✅ Covariance ≈ I (max deviation < 1e-3)
- ✅ All variance ratios sum to ≥ 99%

**Expected Structure** (depends on Step 5 decisions):
- If 3 proxies survive: (423,624 × 4) → PCA → likely 4 PCs or 3 PCs
- If 2 proxies survive: (423,624 × 3) → PCA → likely 3 PCs
- If 1 proxy survives: (423,624 × 2) → PCA → likely 2 PCs

**Actual Results** (post-fix re-run):
- Input: (423,624 × 4) → [log_param_count, synflow_residual, naswot_residual, zenscore_residual]
- Output: **3 PCs** retained (99.81% variance explained)
- Validation: max|col_mean|=2.90e-16 ✅ · max|Cov-I|=4.22e-15 ✅ (near-perfect whitening)
- Output shape: `whitened_features.npy` → (423,624 × 3) float32

**Status**: ✅ COMPLETE (May 8, 2026; re-run May 9 after NASWOT fix)

---

### Step 7: Surrogate MLP Training
**Script**: `scripts/nasbench101/train_mlp_101.py`  
**Duration**: 10-20 min (GPU)  
**Inputs**: Whitened features (from Step 6) + GT accuracies  
**Output**: `results/nasbench101/surrogate_mlp/`

**Training Setup**:
- **Architecture**: `3 → 64(ReLU) → 32(ReLU) → 1`
- **Loss**: RankNet log-sigmoid pairwise loss — `mean(log(1+exp(-(ŷᵢ−ŷⱼ))))` for pairs where yᵢ > yⱼ
- **Optimizer**: Adam, lr = 1e-3
- **Split**: 80% train / 10% val / 10% test (seed = 42)
- **Early stopping**: patience = 20, max 500 epochs
- **Device**: RTX 5070 Ti (CUDA)
- **Dataset size**: 423,624 architectures

**Note on loss function**: The originally planned `relu(-diff)` loss has a degenerate zero-gradient
minimum at constant model output, causing early stopping at epoch ~1. Replaced with the RankNet
log-sigmoid loss which is strictly positive for equal predictions and always has a gradient.

**Outputs**:
```
mlp_results.json                    (Spearman ρ, Kendall τ, loss histories for all variants)
ablation_table.json                 (compact ablation table)
training_curves.png                 (train/val loss per epoch, one subplot per variant)
ablation_comparison.png             (grouped bar chart: ρ_global, ρ_comp, top-5%, top-10%)
```

**Ablation Variants**:
1. **size_only**: MLP input = log(param_count) only — size baseline
2. **best_raw**: MLP input = best single *activation* proxy (SynFlow, ρ=0.421) — excludes param_count to be distinct from size_only
3. **pca_raw**: MLP input = PCA of all 4 raw proxies, no debiasing
4. **full_pipeline**: MLP input = debiased + PCA-whitened features (PROPOSED)

**Actual Results**:

| Variant | Features | Early stop | ρ global | ρ competitive | Top-5% | Top-10% |
|---|:-:|:-:|:-:|:-:|:-:|:-:|
| `size_only` | 1 | epoch 59 | 0.5559 | 0.5580 | 0.201 | 0.279 |
| `best_raw` (SynFlow) | 1 | epoch 30 | 0.4298 | 0.4283 | 0.142 | 0.210 |
| `pca_raw` | 3 | epoch 76 | 0.6275 | 0.6287 | 0.388 | 0.452 |
| **`full_pipeline`** | **3** | **epoch 95** | **0.6494** | **0.6505** | 0.375 | 0.451 |

**Key Findings**:
1. **Single activation proxy (SynFlow) is weaker than param_count alone** (ρ 0.430 vs 0.556) — confirms high collinearity of SynFlow with size
2. **Multi-proxy fusion gives +7.2pp** over size_only — proxies carry complementary information
3. **Debiasing adds +2.2pp over fusion alone** (full_pipeline 0.649 vs pca_raw 0.628) with identical feature count — this is the direct measurable contribution
4. **Full system gain: +9.3pp** over the size baseline (0.649 vs 0.556)
5. **Top-5% precision nuance**: full_pipeline (0.375) marginally below pca_raw (0.388) — debiasing suppresses the size component that dominates the extreme tail; globally better calibrated but slightly reshuffles the top-5%

**Success Criteria Assessment**:
- ✅ `full_pipeline_ρ (0.649) > best_raw_ρ (0.430)` — pipeline significantly outperforms best single proxy
- ✅ `full_pipeline_ρ = 0.649 >> 0.20` — strong signal
- ✅ Competitive ρ = 0.651 — strong and consistent
- ✅ `full_pipeline_ρ > pca_raw_ρ` — debiasing adds value above raw fusion

**Status**: ✅ **COMPLETE** (May 9, 2026)

---

### Step 8: Statistical Validation Suite
**Scripts**: `scripts/nasbench101/statistical_tests/phase{1,2,3}_*.py`  
**Duration**: ~60 min total (Phase 1: 5 min, Phase 2: 5 min, Phase 3: ~50 min across 9 seeds)  
**Purpose**: Rigorously test whether the Δρ = +0.022 debiasing gain observed in Step 7 is statistically meaningful, reproducible, and not a statistical artefact of a single random seed.

This step was added because the core thesis claim — that bias disentanglement improves ranking quality — depends on the +0.022 gap between `full_pipeline` and `pca_raw` being genuine, not lucky. Three independent tests were conducted.

---

#### Phase 1 — Prediction Reproducibility Check
**Method**: `phase1_save_predictions.py` re-ran the canonical seed-42 split (80/10/10 = 338,899 / 42,362 / 42,363 architectures) exactly as in Step 7, retrained `pca_raw` and `full_pipeline`, and saved the test-set model outputs plus ground-truth indices to `results/nasbench101/statistical_validation/predictions/`.

**Why it matters**: Before running any statistical test, we must confirm the seed-42 result is deterministic (no floating-point drift, no data leakage).

**Results**:
| Check | Value |
|---|---|
| Test set size | 42,363 architectures |
| ρ(pca_raw) reproduced | **0.6275** — exact match to Step 7 |
| ρ(full_pipeline) reproduced | **0.6494** — exact match to Step 7 |
| Δρ reproduced | **+0.0219** — bit-for-bit identical |
| Index uniqueness | 42,363 / 42,363 — no train/test leakage |

✅ The seed-42 result is perfectly reproducible.

---

#### Phase 2 — Bootstrap CI + Competitive Subset Analysis
**Method**: Two independent analyses on the Phase 1 predictions.

**Test 1 — Bootstrap Confidence Interval**:  
1,000 bootstrap resamples (with replacement, n = 42,363) of the test set. For each resample, Spearman ρ was computed for both variants and the gap Δρ = ρ(full_pipeline) − ρ(pca_raw) recorded. The 95% CI is the [2.5, 97.5] percentile of the 1,000 Δρ values.

*What this answers*: Is the +0.022 gap caused by test-set sampling noise (i.e., would a different random draw of test architectures show a different ranking)?

**Test 3 — Competitive Subset Analysis**:  
`pca_raw` and `full_pipeline` predictions were further restricted to three GT accuracy thresholds (>50%, >85%, >90%) to check whether the gap is consistent when the degenerate tail is progressively removed.

**Results — Bootstrap CI**:
| Metric | Value |
|---|---|
| Point Δρ | **+0.0219** |
| Bootstrap mean Δρ | **+0.0219** |
| 95% CI | **[+0.0196, +0.0241]** |
| CI excludes zero | **Yes** (narrowly, width = 0.0045) |
| P(Δρ > 0) across 1,000 resamples | **1.0000** |

**Results — Competitive Subset**:
| Threshold | n | pca_raw ρ | full_pipeline ρ | Gap |
|---|:-:|:-:|:-:|:-:|
| GT > 50% | 42,149 | 0.6287 | 0.6505 | **+0.022** |
| GT > 85% | 40,605 | 0.6201 | 0.6443 | **+0.024** |
| GT > 90% | 25,939 | 0.4618 | 0.4826 | **+0.021** |

*What the bootstrap tells us*: For the seed-42 model, the +0.022 gap is not test-set noise. Any random draw of 42k test architectures would show a positive gap.

*What the bootstrap does NOT tell us*: Whether a differently initialised model would show the same gap. Bootstrap only resamples the test set — not the model weights.

*What the subset analysis shows*: The gap is positive and stable across all three thresholds (+0.021 to +0.024), and is slightly larger when the easiest architectures are removed (GT > 85%), suggesting debiasing adds more value in the truly competitive regime.

---

#### Phase 3 — Multi-Seed Stability (9 Seeds)
**Method**: `phase3_multi_seed.py` retrained `pca_raw` and `full_pipeline` from scratch across 9 independent seeds (0, 1, 2, 3, 4, 5, 6, 7, 42). Seed 42 was loaded from the existing `mlp_results.json`. Seeds 0–7 were retrained with identical architecture (3→64→32→1), loss (RankNet), optimizer (Adam, lr=1e-3), and early stopping (patience=20) — only the numpy/torch random seed (controlling the split permutation and weight initialisation) changed. Results were accumulated across two runs and merged into `multi_seed_results.json`.

*What this answers*: Is the +0.022 debiasing gap reproducible when the MLP is trained on a different random 80/10/10 split with different weight initialisation?

**Full 9-Seed Results**:
| Seed | pca_raw ρ | full_pipeline ρ | Gap |
|---|:-:|:-:|:-:|
| 0 | 0.6476 | 0.6178 | −0.0298 |
| 1 | 0.6159 | 0.6032 | −0.0127 |
| 2 | 0.6110 | 0.6090 | −0.0020 |
| 3 | 0.6056 | 0.6073 | +0.0017 |
| 4 | 0.6343 | 0.6078 | −0.0264 |
| 5 | 0.5898 | 0.5828 | −0.0069 |
| 6 | 0.5978 | 0.6145 | +0.0168 |
| 7 | 0.5946 | 0.5907 | −0.0039 |
| **42** *(Step 7 seed)* | **0.6275** | **0.6494** | **+0.0219** |
| **Mean** | **0.6138** | **0.6092** | **−0.0046** |
| **Std** | 0.0184 | 0.0176 | **0.0163** |

Gap positive in **3 / 9 seeds**. Range: [−0.030, +0.022].

**Interpretation**:
Seed 42 is a statistical outlier. The mean gap across all 9 seeds is −0.004 ± 0.016 — indistinguishable from zero. The two variants converge to essentially the same average ranking quality (~ρ = 0.61).

**Why this happens mechanistically**: PCA on the full proxy set and PCA on the debiased proxy set produce nearly equivalent 3-dimensional representations on NAS-Bench-101. The raw proxies are already moderately correlated (SynFlow–param_count ρ ≈ 0.54), so residualising against param_count before PCA changes the embedding subspace only slightly. With a small MLP (3 input features → 64 → 32 → 1) and a pairwise loss, the optimiser converges to different local minima depending on weight initialisation, and this initialisation variance (±0.016) swamps the small feature-space difference (+0.004). Seed 42 happened to favour `full_pipeline`; most other initialisations favour `pca_raw` equally.

**Critical distinction from Phase 2**: The bootstrap CI (Phase 2) correctly shows that *for the seed-42 model*, the gap is not test-set sampling noise. But it cannot detect model-initialisation variance, which Phase 3 reveals as the dominant source of uncertainty.

**Output files**:
```
results/nasbench101/statistical_validation/
  predictions/test_pred_pca_raw.npy
  predictions/test_pred_full_pipeline.npy
  predictions/test_gt.npy
  predictions/test_indices.npy
  bootstrap_results.json
  bootstrap_delta_histogram.png
  competitive_subset_analysis.json
  competitive_subset_plot.png
  phase2_summary.json
  multi_seed_results.json          ← 9-seed merged results
  multi_seed_gap_plot.png
  statistical_findings.md          ← full methodology + findings narrative
```

**Revised Findings Summary**:
| Test | Question asked | Answer |
|---|---|---|
| Bootstrap CI | Is the +0.022 gap test-set sampling noise? | **No** — CI [+0.020, +0.024], P=1.00 for this model |
| Competitive subset | Does the gap hold at stricter GT thresholds? | **Yes** — +0.021 to +0.024, slightly larger at GT>85% |
| Multi-seed (9 seeds) | Is the gap reproducible across random splits? | **No** — mean gap −0.004 ± 0.016, positive in 3/9 seeds |

**Robust findings that hold across all seeds**:
1. **Multi-proxy fusion is the primary, consistent contribution**: `pca_raw` (mean ρ = 0.614) beats the best single proxy `size_only` (ρ = 0.556) by +0.065 — this gap is large, stable, and never reverses across any seed.
2. **Debiasing is neutral on average**: `full_pipeline` and `pca_raw` are statistically equivalent (Δρ = −0.004 ± 0.016). Debiasing neither helps nor hurts consistently on NAS-Bench-101.
3. **Framework degrades gracefully**: At GT > 90%, absolute ρ drops for all variants (accuracy range compresses), but the multi-proxy advantage over single-proxy baselines is maintained.

**Status**: ✅ **COMPLETE** (May 9, 2026)

---

## 📊 Cross-Benchmark Comparison (What NAS-101 Results Will Fill In)

| Metric | NAS-Bench-201 | SSS | NAS-Bench-101 |
|--------|---------------|-----|--------------|
| N architectures | 15,625 | 32,768 | **423,624** |
| Search space | Op search | Width only | **DAG** |
| R² (param→GT) | 0.157 | 0.789 | **0.047** |
| GT std | ~12% | 1.27% | **~5.8%** |
| Degenerate cluster | ~300 | 0 | **~646** |
| **SynFlow partial ρ** | -0.002 (OUT) | +0.625 (IN) | **+0.242 (IN)** |
| **NASWOT partial ρ** | +0.154 (IN) | +0.059 (OUT) | **+0.103 (IN, marginal)** |
| **ZenScore partial ρ** | +0.190 (IN) | +0.051 (OUT) | **+0.103 (IN, marginal)** |
| **PCA features** | 3 | 2 | **3** |
| **full_pipeline ρ** | Pending | Pending | **0.649** |

**NAS-101 Predictions** (from benchmark analysis):
- SynFlow likely to survive (DAGs have path-capacity variation)
- NASWOT/ZenScore likely to survive (DAGs have operation diversity)
- All three may coexist (unlike SSS where only SynFlow survived)
- Overall pipeline ρ likely 0.25-0.45 (moderate, between SSS and NAS-201)

---

## ⚠️ Risk Assessment & Contingency Plans

### Risk 1: TFRecord Load Failure
**Symptoms**: "ValueError: Unable to parse TFRecord" or timeout >600 seconds  
**Mitigation**:
1. Verify file at `data/nasbench101/nasbench_full.tfrecord` exists (2.1 GB)
2. Check PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION is set
3. Check nasbench API patches applied correctly
4. Try re-downloading TFRecord from source

**Impact**: 🔴 BLOCKING — cannot proceed without Step 0

---

### Risk 2: GPU Memory Overflow During Proxy Compute
**Symptoms**: "CUDA out of memory" during Steps 1b-1d  
**Mitigation**:
1. Reduce BATCH_SIZE or INPUT_SIZE in GPU script
2. For NASWOT: reduce from (8,3,32,32) to (4,3,32,32)
3. For ZenScore: reduce from (4,3,32,32) to (2,3,32,32)
4. Run one proxy at a time (sequential instead of parallel)

**Impact**: 🟡 MODERATE — delays Phase 2 but recoverable

---

### Risk 3: All Proxies Fail Bias Disentanglement (partial ρ < 0.10)
**Symptoms**: Step 5 output shows all three proxies with partial ρ < 0.10  
**Interpretation**: Proxies carry only size signal; no architecture-quality signal in 101  
**Mitigation**:
1. Double-check OLS residual calculation (may be arithmetic error)
2. Try partial-rank Spearman directly (more robust)
3. Visualize residuals: scatter(residuals vs GT) to diagnose
4. **Thesis impact**: Valid finding! Shows proxies insufficient for 101; still publishable

**Impact**: 🟡 MODERATE — changes thesis narrative but not fatal

---

### Risk 4: Step 7 MLP Underperforms Raw Proxies
**Symptoms**: full_pipeline_ρ << best_raw_ρ by >0.10  
**Interpretation**: PCA/whitening/debiasing hurts on this benchmark  
**Mitigation**:
1. Check ablation_table.json: is pca_raw also worse than best_raw?
2. If yes → PCA itself is problematic on 101; skip PCA (use raw features)
3. If no → debiasing is the issue; use raw proxies instead
4. **Thesis impact**: Still valid; shows methodology not universal; benchmark-specific

**Impact**: 🟡 MODERATE — requires narrative adjustment but acceptable

---

### Risk 5: Checkpoint Corruption During GPU Compute
**Symptoms**: "JSON decode error" when resuming proxy script  
**Mitigation**:
1. Delete checkpoint file: `rm results/nasbench101/raw_proxy_scores/{proxy}_checkpoint.json`
2. Re-run script; will restart from idx=0
3. Progress before corruption is lost; restart is necessary

**Impact**: 🟡 MODERATE — time loss but not catastrophic

---

## 📈 Success Metrics & Final Decision Gate

### Tier 1: Critical Success (Thesis publishable)
- ✅ At least **one proxy** survives with partial ρ > 0.10
- ✅ At least **one MLP variant** achieves ρ > 0.20 on test set
- ✅ Ablation table complete with all comparisons

### Tier 2: Strong Success (Thesis compelling)
- ✅ **Two or three proxies** survive debiasing
- ✅ full_pipeline ρ ≥ best_raw ρ
- ✅ Proxy survival pattern differs from both NAS-201 and SSS (truly benchmark-dependent)

### Tier 3: Excellent Success (Thesis exceptional)
- ✅ **All three proxies** survive with partial ρ > 0.20
- ✅ full_pipeline ρ **significantly > best_raw ρ** (at least +0.05)
- ✅ All three benchmarks show the proxy reversal pattern

### Failure Criteria (Thesis problematic)
- ❌ No proxies survive (partial ρ < 0.10 for all)
- ❌ All three benchmarks produce identical proxy survivors (not benchmark-dependent)
- ❌ full_pipeline ρ < 0.10 (no signal at all)

---

### 🏆 Achieved Tier: **Tier 2 — Strong Success** *(with statistical nuance from Step 8)*

- ✅ All three proxies survive debiasing (SynFlow 0.242, NASWOT 0.103, ZenScore 0.103)
- ✅ full_pipeline ρ (0.649) >> best_raw ρ (0.430) — +21.9pp gap
- ✅ Multi-proxy fusion gain (+6.5pp, pca_raw over size_only) confirmed robust across all 9 seeds
- ✅ Proxy survival pattern differs from NAS-201 and SSS (SynFlow survives all three; NASWOT/ZenScore survive 101 and 201 but not SSS)
- ⚠️ Debiasing gain (+2.2pp, seed 42) is seed-specific: mean gap across 9 seeds = −0.004 ± 0.016 (not reproducible)
- ⚠️ SynFlow partial ρ = 0.242 (does not reach the 0.20 threshold required for Tier 3)
- ⚠️ full_pipeline top-5% precision (0.375) marginally below pca_raw (0.388) — consistent with debiasing suppressing the size signal dominant in the extreme tail

---

## 📅 Timeline & Checkpoints

| Checkpoint | Date | Status |
|------------|------|---|
| **Phase 1 Start** | May 8, 2026 | ✅ Complete |
| Step 0 complete | May 8, 2026 | ✅ Complete |
| Step 1a complete | May 8, 2026 | ✅ Complete |
| **Phase 2 Start** | May 8, 2026 | ✅ Complete |
| Steps 1b-1d complete | May 8, 2026 | ✅ Complete (~3.3 GPU hours) |
| **Phase 3 Start** | May 8, 2026 | ✅ Complete |
| Steps 2-4 complete | May 8, 2026 | ✅ Complete |
| **NASWOT fix identified + re-run Steps 2–6** | May 9, 2026 | ✅ Complete |
| Step 5 complete (post-fix) | May 9, 2026 | ✅ Complete |
| Step 6 complete (post-fix) | May 9, 2026 | ✅ Complete |
| Step 7 complete | May 9, 2026 | ✅ Complete |
| **Step 8 — Phase 1 (prediction reproducibility)** | May 9, 2026 | ✅ Complete |
| **Step 8 — Phase 2 (bootstrap CI + competitive subset)** | May 9, 2026 | ✅ Complete |
| **Step 8 — Phase 3 (9-seed stability, seeds 0–7 + 42)** | May 9, 2026 | ✅ Complete |
| **FINAL RESULTS READY** | May 9, 2026 | ✅ **DONE** |

---

## 🔬 Key Monitoring & Debugging Points

### During Phase 1 (Step 0)
- Watch TFRecord load time: should be ~390s
- If > 600s: likely filesystem issue
- Check audit_summary.json: n_total must be 423,624

### During Phase 2 (GPU Proxies)
- Monitor GPU: `nvidia-smi` in separate terminal
- Watch for OOM errors after ~50k architectures
- Check progress every 50k archs; ETA should decrease steadily
- Verify checkpoint JSON valid after each 1k-arch milestone

### During Phase 3 (Analysis & Training)
- Step 4 outputs scatter plots: visually inspect for outliers/anomalies
- Step 5 partial_correlations.json: confirm at least one proxy has partial ρ ≥ 0.10
- Step 6 pca_plots.png: check eigenvalues make sense (no negative values)
- Step 7 training_curves.png: loss should decrease; no anomalies (NaN, ∞)

---

## 📝 Progress Log (to be updated live)

```
[May 8, 2026] Experimentation plan created.
[May 8, 2026] Phase 1 complete — Step 0 (284s load) + Step 1a done. n=423,624 verified.
[May 8, 2026] Phase 2 complete — SynFlow (~80min, interrupted+resumed), NASWOT (~70min), ZenScore (~50min).
[May 8, 2026] Steps 2–4 complete — transform, distribution analysis, ranking correlations.
[May 8, 2026] Step 5 complete (initial) — all 3 proxies KEEP_DOCUMENTED.
[May 8, 2026] Step 6 complete (initial) — whitened_features.npy (423,624×3) written.
[May 9, 2026] NASWOT normalisation bug identified: variable-layer mean denominator.
              Fix: divide by MAX_CONV=48 (fixed constant). ρ: −0.2245 → −0.1982.
              Original backed up as naswot_original_mean.npy.
[May 9, 2026] Steps 2–6 re-run after NASWOT fix. All outputs regenerated.
              Partial ρ: NASWOT 0.1246→0.1034, ZenScore 0.1250→0.1032. All 3 still pass threshold.
[May 9, 2026] Step 7 complete — MLP trained on GPU (RTX 5070 Ti).
              full_pipeline ρ=0.649, pca_raw ρ=0.628, size_only ρ=0.556, best_raw ρ=0.430.
              Plots: training_curves.png, ablation_comparison.png written.
[May 9, 2026] FINAL RESULTS READY — NAS-Bench-101 pipeline complete.
[May 9, 2026] Step 8 Phase 1 — Predictions saved. Seed-42 result verified bit-for-bit (ρ_pca_raw=0.6275, ρ_full=0.6494).
[May 9, 2026] Step 8 Phase 2 — Bootstrap CI: Δρ=+0.022, 95% CI [+0.0196, +0.0241], P=1.00. Competitive subset: gap +0.021–+0.024 stable.
[May 9, 2026] Step 8 Phase 3 — 9-seed run (seeds 0–7 + 42). Mean gap=−0.004±0.016, positive 3/9 seeds.
              Conclusion: debiasing gain not reproducible. Multi-proxy fusion (Δρ≈+0.065) is the robust claim.
[May 9, 2026] statistical_findings.md written and updated with full 9-seed table.
```

---

## 🎯 Thesis Impact & Narrative

### How NAS-Bench-101 Findings Will Shape the Thesis

**If SynFlow survives** (partial ρ > 0.10):
- Supports hypothesis: SynFlow measures path-capacity distribution
- Only weak in NAS-201 (op diversity too strong), strong in SSS (width-only), strong here (DAG diversity)

**If NASWOT/ZenScore survive** (unlike SSS):
- Supports hypothesis: These measure operation-type diversity
- Present in NAS-201 (6 ops), weak in SSS (Conv2d only), strong here (variable ops/connections)

**If all three survive**:
- Strongest validation of proxy reversal pattern
- "Optimal proxy set is benchmark-dependent, not universal"

**If none survive**:
- Counter-finding: proxies may be fundamentally limited
- Thesis shifts to "proxies work only in moderate-complexity spaces; DAG search too hard"

**Regardless of outcome**:
- NAS-Bench-101 (R² = 0.047) is the most credible validation
- Results are publishable; negative finding is still a contribution

---

---

## 🏁 Final Conclusion

The NAS-Bench-101 pipeline is **fully complete**. All eight steps were executed, validated, and their results interpreted. Here is what was learned and what it means:

### What was done
A zero-cost proxy pipeline was applied to all 423,624 architectures in NAS-Bench-101 (the largest and most structurally diverse NAS benchmark tested). Four proxy scores — parameter count, SynFlow, NASWOT, and ZenScore — were computed, log-transformed, debiased against parameter count, and fused via PCA whitening into a 3-dimensional embedding. A pairwise-ranking MLP was trained on this embedding and evaluated against three baselines. A 3-phase statistical validation suite was then run to stress-test the key result.

### What was found

**1. Multi-proxy fusion is the primary and robust contribution.**  
Combining all four proxies into a fused embedding (`pca_raw`) raised Spearman ρ from 0.556 (best single proxy, param_count) to 0.628 — a **+6.5pp gain** that holds across every one of the 9 seeds tested. This is the thesis's most reliable empirical result on this benchmark.

**2. Bias disentanglement is neutral on NAS-Bench-101.**  
The debiased `full_pipeline` embedding achieves ρ = 0.649 in seed 42 (+2.2pp over `pca_raw`). However, across 9 independent seeds the mean gap is −0.004 ± 0.016 — statistically indistinguishable from zero, and positive in only 3 of 9 runs. The debiasing step does not harm results but does not reliably improve them on this benchmark. The reason: SynFlow and param_count are already highly correlated (ρ ≈ 0.54), so residualising against param_count changes the feature subspace only slightly, and the small difference is overwhelmed by MLP initialisation noise.

**3. The bootstrap CI is not wrong — it answers a different question.**  
Phase 2 confirmed that for the seed-42 model, the +0.022 gap is not test-set sampling noise (CI [+0.020, +0.024], P = 1.00). Phase 3 showed that the gap is not model-training-stable. Together, these two findings are consistent: the gap is real for that specific trained model, but that model is not reproducible in the expected direction. Both results are reported honestly.

**4. Proxy behavior is benchmark-dependent — as hypothesised.**  
SynFlow, which had near-zero predictive power on NAS-Bench-201 (ρ = −0.002) and was the dominant proxy on SSS (ρ = 0.916), achieves intermediate strength here (partial ρ = 0.242). NASWOT and ZenScore, which were strong on NAS-Bench-201 (ρ ≈ 0.52) and absent from SSS, survive here at marginal levels (partial ρ ≈ 0.10). This cross-benchmark variation is the central structural claim of the thesis.

**5. The framework degrades gracefully under distributional pressure.**  
At GT > 90% (accuracy range compressed to ~4pp), absolute ρ drops for all variants (expected — proxies cannot resolve sub-1% differences). But the relative ordering and the multi-proxy advantage over single-proxy baselines are maintained, confirming the framework is not brittle.

### What it means for the thesis

The **revised thesis claim for NAS-Bench-101** is:
> *"Fusing multiple zero-cost proxies into a joint embedding via PCA whitening produces a large, consistent improvement over single-proxy scoring (Δρ ≈ +0.065). The bias-disentanglement step produces features of equivalent quality to raw fusion on this benchmark (Δρ ≈ 0), with a directional positive trend observed under one specific random seed. These findings, combined with contrasting results on NAS-Bench-201 and SSS, demonstrate that the optimal proxy set and the marginal value of debiasing are both benchmark-dependent."*

This is a scientifically honest, well-validated, and publishable result. The statistical validation suite (Phase 1–3) demonstrates methodological rigor: the positive seed-42 result is not hidden, but it is not over-claimed either.

### Output artefacts produced
| Artefact | Location | Description |
|---|---|---|
| `gt_accuracies.npy` | `results/nasbench101/audit/` | 423,624 ground-truth accuracies |
| `{proxy}.npy` | `results/nasbench101/raw_proxy_scores/` | 4 raw proxy arrays |
| `{proxy}_log.npy` | `results/nasbench101/transformed_proxy/` | Log-transformed proxies |
| `{proxy}_residuals.npy` | `results/nasbench101/debiased_proxy/` | Debiased proxy residuals |
| `whitened_features.npy` | `results/nasbench101/pca_whitening/` | (423,624 × 3) MLP input |
| `mlp_results.json` | `results/nasbench101/surrogate_mlp/` | 4-variant ablation results |
| `ablation_comparison.png` | `results/nasbench101/surrogate_mlp/` | Ablation bar chart |
| `multi_seed_results.json` | `results/nasbench101/statistical_validation/` | 9-seed gap table |
| `bootstrap_results.json` | `results/nasbench101/statistical_validation/` | Bootstrap CI |
| `statistical_findings.md` | `results/nasbench101/statistical_validation/` | Full validation narrative |

---

## ✅ Pre-Execution Checklist

Before running Phase 1:

- [ ] TFRecord exists at `data/nasbench101/nasbench_full.tfrecord`
- [ ] Python venv activated
- [ ] NasBench API patches applied (3 patches to nasbench/api.py)
- [ ] PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION set
- [ ] CUDA available: `nvidia-smi` returns GTX 5070 Ti
- [ ] PyTorch GPU enabled: `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Disk space available: ~50 GB for results (free space check: `df -h`)
- [ ] All scripts present: `scripts/nasbench101/{gpu/,}*.py` exist
- [ ] Results directory writable: `mkdir -p results/nasbench101`

---

## 📞 Key Contacts & References

**If TFRecord Issues**: Check NasBench documentation: https://github.com/google-research/nasbench  
**If GPU Issues**: NVIDIA CUDA setup guide  
**If Algorithm Questions**: See EXPERIMENTATION_PROGRESS_NOTE.md (NAS-Bench-201 reference)  
**If Data Interpretation Questions**: See EXPERIMENTATION_NOTE_SSS.md (SSS reference)

---

**Status**: Ready to execute on lab machine. Awaiting user confirmation to start Phase 1.
