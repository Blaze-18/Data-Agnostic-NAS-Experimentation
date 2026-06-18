# Bias-Disentangled Structural Proxy Embedding Framework for Data-Agnostic NAS

**Thesis Project**: Zero-Cost Proxy-Based Architecture Ranking Without Training Data

---

## 🎯 Core Framework

A **data-agnostic surrogate model** that ranks neural architectures using only structural zero-cost proxies, with size bias removed via disentanglement.

### Five-Stage Pipeline

1. **Structural Proxy Extraction** → Compute 4 proxies at initialization: SynFlow, Zen-Score, NASWOT, Param Count
2. **Bias Disentanglement** → Regress out structural size factors via OLS; retain residuals as bias-corrected signals
3. **Proxy Embedding Construction** → Apply PCA + whitening to orthogonalize and normalize features
4. **Surrogate Training** → Shallow MLP trained with pairwise ranking loss (optimizes rank ordering, not point accuracy)
5. **Data-Agnostic Inference** → At search time: extract proxies → apply transformations → MLP scores (zero dataset access)

**Key Property**: All proxies computed on random inputs; no training data required at any stage.

---

## 📊 Benchmark Status & Results

### Completed Benchmarks

#### **NAS-Bench-201** (15,625 architectures)
| Property | Value |
|----------|-------|
| Search space | Operation search (6 ops, 6 edges) |
| Size bias (R²) | 0.157 — size moderately relevant |
| GT accuracy range | ~10% – 94.4% |
| Degenerate cluster | ~300 architectures (failed to train) |
| **Status** | ✅ Steps 0-6 COMPLETE |

**Proxy survival post-debiasing**:
- ✅ **NASWOT**: partial ρ = +0.154 (KEPT)
- ✅ **ZenScore**: partial ρ = +0.190 (KEPT)
- ❌ **SynFlow**: partial ρ = -0.002 (EXCLUDED)
- **PCA input**: param_count + NASWOT_residual + ZenScore_residual (3 features)

#### **NATS-Bench SSS** (32,768 architectures)
| Property | Value |
|----------|-------|
| Search space | Width-only (8 channel widths, fixed topology) |
| Size bias (R²) | 0.789 — size dominates (benchmark limitation) |
| GT accuracy range | 79.79% – 93.65% |
| Degenerate cluster | None (all architectures train successfully) |
| **Status** | ✅ Steps 0-6 COMPLETE |

**Proxy survival post-debiasing**:
- ✅ **SynFlow**: partial ρ = +0.625 (KEPT, strong)
- ❌ **NASWOT**: partial ρ = +0.059 (EXCLUDED)
- ❌ **ZenScore**: partial ρ = +0.051 (EXCLUDED)
- **PCA input**: param_count + SynFlow_residual (2 features)

#### **NAS-Bench-101** (423,624 architectures) — PRIMARY BENCHMARK
| Property | Value |
|----------|-------|
| Search space | DAG search (variable 4-7 node DAGs) |
| Size bias (R²) | **0.047** — size nearly irrelevant (BEST test case) |
| GT accuracy range | ~10% – 94.32% |
| Degenerate cluster | ~646 architectures (<20% accuracy) |
| **Status** | ❌ Steps 0-7 PENDING — PRIORITY |

**Why NAS-Bench-101 matters**: With R² = 0.047, proxies must survive debiasing by extracting genuine architecture quality signal, not size. This is the hardest test and most credible result for the thesis.

### Key Finding: Proxy Behavior Reversal

Proxies show **benchmark-dependent survival patterns**:

| Proxy | NAS-Bench-201 | SSS | NAS-Bench-101 |
|-------|---------------|-----|--------------|
| **SynFlow** | ❌ Noise (-0.002) | ✅ Strong (+0.625) | **?** |
| **NASWOT** | ✅ Kept (+0.154) | ❌ Noise (+0.059) | **?** |
| **ZenScore** | ✅ Kept (+0.190) | ❌ Noise (+0.051) | **?** |

**Interpretation**:
- **SynFlow** measures path-capacity distribution (powerful in pure-width spaces, weak in operation-diverse spaces)
- **NASWOT/ZenScore** measure operation-type diversity (powerful when ops vary, weak when all Conv2d)
- **NAS-Bench-101 prediction**: All three may survive because DAGs have both operation diversity AND variable path capacity

---

## 💾 Current Implementation Status

### ✅ Completed (Steps 0-6 on NAS-Bench-201 & SSS)

- ✅ Proxy computation (all 4 proxies × 2 benchmarks)
- ✅ Log-transformation with automatic sign detection
- ✅ Distribution analysis & ranking correlation validation
- ✅ Bias disentanglement via OLS + partial-rank Spearman correlation
- ✅ PCA whitening with identity covariance validation
- ✅ Feature matrices ready for MLP training

### ⏳ In Progress / TODO

| Task | Status | Timeline |
|------|--------|----------|
| **NAS-Bench-101 Steps 0-6** | ❌ NOT STARTED | Priority — lab machine |
| **NAS-Bench-201 Step 7 (MLP)** | ❌ NOT STARTED | After NAS-Bench-101 audit |
| **SSS Step 7 (MLP)** | ❌ NOT STARTED | After NAS-Bench-101 audit |
| **Thesis 3-benchmark table** | ⏳ Pending | All MLP trainings done |

### Key Dependencies

1. **NAS-Bench-101 Step 0** must complete successfully (loads TFRecord ~390 seconds)
2. **NAS-Bench-101 Steps 1b-1d** run in parallel on GPU (SynFlow, NASWOT, ZenScore)
3. **NAS-Bench-101 Step 7** uses output from Step 6 (whitened features + MLP training)
4. **Final comparison table** requires all three MLP trainings complete

---

## 📋 Pipeline Steps Detail

### Step 0: Structural Audit
- Load TFRecord; extract GT accuracies + param counts
- Verify data integrity and distribution statistics
- Identify degenerate architectures (failed training)
- **Time**: ~390 seconds (one-time load)

### Steps 1a-1d: Proxy Computation
- **1a**: Param count (trivial, copied from audit)
- **1b**: SynFlow (GPU, ~15-30 min) — Tanaka et al. 2020 algorithm
- **1c**: NASWOT (GPU, ~15-30 min) — multi-layer Conv2d hooking
- **1d**: ZenScore (GPU, ~20-40 min) — 4× averaged covariance trace
- **Note**: Run 1b-1d in parallel; all write to different output files

### Steps 2-4: Analysis
- **Step 2**: Log-transform proxies (auto-detect negation if needed)
- **Step 3**: Distribution analysis (statistics, normality tests)
- **Step 4**: Ranking correlation validation (Spearman ρ, Kendall τ vs GT)

### Step 5: Bias Disentanglement
- OLS regression: `Proxy ~ param_count`
- Compute residuals (bias-corrected signal)
- Partial-rank Spearman correlation to remove discrete covariate effects
- Decision: retain proxies with partial ρ ≥ 0.10

### Step 6: PCA Whitening
- Standardize features (z-score)
- Eigen-decomposition of covariance matrix
- Retain PCs with cumulative variance ≥ 99%
- Whiten output (enforce identity covariance)
- Validate: column means ≈ 0, covariance = I

### Step 7: Surrogate MLP Training
- **Architecture**: input → 64(ReLU) → 32(ReLU) → 1
- **Loss**: Pairwise ranking loss (optimizes rank ordering, not point accuracy)
- **Split**: 80% train / 10% val / 10% test
- **Evaluation**: Spearman ρ, Kendall τ, top-K precision
- **Ablation**: raw features vs PCA-only vs full pipeline

---

## 🔧 Executing the Full Pipeline

### NAS-Bench-101 on Lab Machine (GTX 5070 Ti)

**Step 0: Audit**
```bash
python scripts/nasbench101/step0_audit_101.py
```
Loads TFRecord (~390 seconds), outputs to `results/nasbench101/audit/`

**Steps 1b-1d: Proxy Computation (run in parallel in separate terminals)**
```bash
python scripts/nasbench101/gpu/compute_proxy_synflow_gpu.py
python scripts/nasbench101/gpu/compute_proxy_naswot_gpu.py
python scripts/nasbench101/gpu/compute_proxy_zenscore_gpu.py
```
Each checkpoints every 1000 architectures; can resume if interrupted.

**Steps 2-7: Analysis & Training (sequential)**
```bash
python scripts/nasbench101/transform_proxies_101.py
python scripts/nasbench101/analyze_distributions_101.py
python scripts/nasbench101/validate_ranking_correlations_101.py
python scripts/nasbench101/bias_disentanglement_101.py
python scripts/nasbench101/pca_whitening_101.py
python scripts/nasbench101/train_mlp_101.py
```
Total time: Steps 2-7 combined ~20-30 minutes.

---

## 📁 Output Structure

Results are organized by benchmark:

```
results/
├── nasbench101/
│   ├── audit/                    # Step 0: GT + param counts
│   ├── raw_proxy_scores/         # Steps 1a-1d: raw proxy values
│   ├── transformed_proxy/        # Step 2: log-transformed
│   ├── proxy_distribution/       # Step 3: statistics
│   ├── proxy_validation/         # Step 4: correlations vs GT
│   ├── debiased_proxy/           # Step 5: residuals + partial rho
│   ├── pca_whitening/            # Step 6: whitened features
│   └── surrogate_mlp/            # Step 7: model + ablation results
├── nasbench201/
│   ├── ... (same structure)
│   └── surrogate_mlp/            # Step 7: MLP results
└── nats_bench_sss/
    ├── ... (same structure)
    └── surrogate_mlp/            # Step 7: MLP results
```

---

## ✅ Success Criteria (Thesis Acceptance)

For each benchmark, results are acceptable if:

1. **At least one proxy survives debiasing** with partial ρ > 0.10
2. **Pipeline outperforms single best proxy**: `full_pipeline_ρ ≥ best_raw_proxy_ρ`
3. **Clear positive signal**: `full_pipeline_competitive_ρ > 0` (on non-degenerate architectures)

**Note**: Even if pipeline underperforms raw proxies, it's a valid finding—shows benchmark-specific limitations of the debiasing approach.

---

## 🏆 Thesis Narrative

**Central question**: Can zero-cost proxies rank architectures fairly after removing structural size bias?

**Benchmark progression**:
- **NAS-Bench-201** (R² = 0.157): Can proxies work when size is moderately relevant?
- **SSS** (R² = 0.789): Can proxies work when size dominates? (stress test)
- **NAS-Bench-101** (R² = 0.047): Can proxies work when size is nearly irrelevant? (litmus test)

**Proxy survival reversal** shows that optimal proxy sets are benchmark-dependent, not universal.

---

## 📊 Key Files for Thesis

After all runs complete, these files contain the core results:

```
results/nasbench101/audit/audit_summary.json              → Size bias quantification
results/nasbench101/proxy_validation/correlation_results.json  → Raw proxy signals
results/nasbench101/debiased_proxy/partial_correlations.json   → Debiasing effectiveness
results/nasbench101/surrogate_mlp/ablation_table.json     → Pipeline comparison

(same files for nasbench201/ and nats_bench_sss/)
```

---

## 🖥️ Environment Setup

### Lab Machine (NAS-Bench-101)
- **Python**: 3.12
- **PyTorch**: GPU-enabled (CUDA 12.8)
- **Key packages**: numpy, scipy, scikit-learn, tensorflow, nasbench, matplotlib
- **GPU**: GTX 5070 Ti (confirmed via nvidia-smi)
- **TFRecord**: `data/nasbench101/nasbench_full.tfrecord` (2.1 GB)

### Result Aggregation
After each benchmark completes, copy key JSON files to version control:
```bash
git add results/*/audit/audit_summary.json
git add results/*/proxy_validation/correlation_results.json
git add results/*/debiased_proxy/partial_correlations.json
git add results/*/pca_whitening/pca_summary.json
git add results/*/surrogate_mlp/ablation_table.json
```

These are small (<50 KB each) and enable cross-machine result tracking.

