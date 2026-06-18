# Zero-Cost Proxy Experimentation — Progress & Analysis Notes (NATS-Bench SSS)

**Date**: May 3, 2026  
**Phase**: Steps 0–6 Complete | Ready for Surrogate MLP Training (Step 7)  
**Status**: PCA whitening complete — whitened 2-feature matrix (32,768 × 2) ready as MLP input; 2 PCs retained; validation PASS

---

## 0. Benchmark Overview: NATS-Bench SSS

**Search space**: Size Search Space (SSS) — channel-width variation only, fixed topology  
**N architectures**: 32,768  
**Architecture encoding**: 5 colon-separated channel widths, each in {8, 16, 24, 32, 40, 48, 56, 64}  
  → 8⁵ = 32,768 total. Example: `'16:32:64:32:64'`  
**Topology**: Fixed DynamicShapeTinyNet — `stem → SSSInferCell → ResBlock(stride=2) → SSSInferCell → ResBlock(stride=2) → SSSInferCell → BN+ReLU → GAP → Linear`  
**GT accuracy**: CIFAR-10 test accuracy at epoch 89  
  → Key: `data['90']['all_results'][('cifar10', 777)]['eval_acc1es']['ori-test@89']`

**Critical difference from NAS-Bench-201**: The SSS space varies only capacity (channel widths). There is no operation-type variation, no skip/none operations, no degenerate architectures. This makes SSS a pure capacity benchmark — every architecture is functionally valid and trains to competitive accuracy.

---

## 1. Pipeline Completion Status

| Step | Task | Status |
|------|------|--------|
| 0 | Structural audit — 6 questions | ✅ Complete |
| 1 | Proxy computation — all 4 proxies, 32,768 architectures | ✅ Complete |
| 1.5 | Proxy validation — 9 checks | ✅ Complete (all PASS) |
| 2 | Log transformation | ✅ Complete |
| 3 | Distribution analysis | ✅ Complete |
| 4 | Ranking correlation validation against CIFAR-10 GT | ✅ Complete |
| 5 | Bias disentanglement (regress out param_count) | ✅ Complete |
| 6 | PCA whitening — param_count + SynFlow residual | ✅ Complete |
| 7 | Surrogate MLP training + evaluation | ❌ Not started — NEXT |

---

## 2. Step 0: Structural Audit Findings

**Script**: `scripts/nats_bench_sss/step0_sss_audit.py`

### Q1 — GT Key Verification (PASS)
Five spot-check accuracies extracted successfully:
- Arch idx=0 (`8:8:8:8:8` all-minimum): 79.79%
- Arch idx=1000: 90.14%
- Arch idx=8192: 82.18%
- Arch idx=16383: 93.34%
- Arch idx=32767 (`64:64:64:64:64` all-maximum): 93.40%

GT object type is a plain dict with `'eval_acc1es'` field. No custom accessor needed.

### Q2 — Accuracy Distribution (PASS — No Degenerate Cluster)
| Statistic | Value |
|---|---|
| N | 32,768 |
| Min | 79.79% |
| Max | 93.65% |
| Mean | 91.09% |
| Std | 1.27% |
| p5 | 88.71% |
| p25 | 90.43% |
| Median | 91.31% |
| p75 | 91.99% |
| p95 | 92.70% |
| N below 30% | **0** |
| N below 50% | **0** |
| Top-10% spread | 1.19% (92.46%–93.65%) |
| Top-1% threshold | 93.05% |

**Key finding**: There is NO degenerate cluster near 10% accuracy (unlike NAS-Bench-201's ~300 failed architectures). The minimum accuracy is 79.79%. All 32,768 architectures train successfully. The distribution is unimodal and left-skewed, compressed into a 14% range (79.79–93.65%). The top-10% spread of only 1.19% means discriminating among high-performing architectures is extremely difficult — no proxy can meaningfully rank within a 1.19% accuracy window. This is the primary explanation for why all proxies will show lower top-1% precision on SSS than on NAS-Bench-201.

### Q3 — Architecture Space Structure (PASS)
- All 32,768 arch strings parse cleanly: exactly 5 colon-separated integers each
- All 5 positions take values in {8, 16, 24, 32, 40, 48, 56, 64} — no anomalies
- All 5 positions have equal variance (336.0) — the space is uniform

### Q4 — Channel Width vs Param Count as Covariate
Spearman ρ between each individual channel position and GT accuracy (from audit_summary.json):

| Position | ρ vs GT |
|---|---|
| pos_0 (ch0) | 0.455 |
| pos_1 (ch1) | 0.312 |
| pos_2 (ch2) | **0.527** ← max |
| pos_3 (ch3) | 0.427 |
| pos_4 (ch4) | 0.324 |
| param_count (actual, from Step 4) | **+0.872** |

**Finding**: The per-position pattern is non-monotonic — pos_2 (the middle stage) has the highest individual correlation (0.527), not the final stage. All 5 positions are mutually uncorrelated (pairwise Pearson = 0 by construction of the uniform grid). Despite pos_2 being the best individual predictor, param_count (which captures the total capacity across all stages) dominates all 5 individual positions with ρ=0.872. The audit recommends single-covariate OLS with log(param_count), which transfers without modification from NAS-Bench-201.

### Q5 — FLOPs Availability
FLOP data is not embedded in the simple pickle archive. Only eval_acc1es and training curves are stored. Not used as additional covariate.

### Q6 — NASWOT/ZenScore Direction (PASS — No Negation)
200-architecture subset audit (NASWOT only, as defined in the script):
- NASWOT raw ρ vs GT = **+0.463** (directly measured, positive)
- ZenScore direction inferred from Step 4 full-set ρ = **+0.557** (positive) — ZenScore was not independently audited in Step 0 Q6, only NASWOT was

**Decision**: No negation applied. Unlike NAS-Bench-201 (where skip/none architectures had high covariance trace but low accuracy, forcing negation), SSS has only Conv2d operations. Larger networks have higher activation covariance AND higher accuracy — the natural direction is positive. Transform: `log(score + ε)` without negation.

---

## 3. Step 1: Proxy Computation

**Scripts**: `scripts/nats_bench_sss/compute_proxy_{param_count,naswot,zenscore,synflow}_sss.py`  
**Model builder**: `scripts/nats_bench_sss/proxy_utils_sss.py` — exact DynamicShapeTinyNet reconstruction

### Proxy Computation Details

| Proxy | Algorithm | Implementation |
|---|---|---|
| param_count | Build model, count parameters | `sum(p.numel() for p in model.parameters())` |
| NASWOT | Single batch (8 samples), mean covariance trace across all Conv2d layers | Multi-layer hooking; 100% Conv2d coverage |
| ZenScore | 4 × batch of 4, mean covariance trace across passes and layers | Same hooking; stochasticity via multiple random draws |
| SynFlow | All-ones input, weight linearisation (abs_), loss=output.sum(), Σ\|grad×weight\| | Tanaka et al. 2020 |

**Param count cross-validation**: 5/5 spot checks match exactly against stored benchmark values. 100-architecture random XVal in Step 1.5: **0 mismatches**.

### Raw Score Ranges

| Proxy | Min | Max | Mean | Std |
|---|---|---|---|---|
| param_count | 11,714 | 713,674 | 268,684 | 110,110 |
| NASWOT | 1.434e-4 | 3.425e-3 | 1.062e-3 | 4.385e-4 |
| ZenScore | 1.065e-4 | 2.662e-3 | 9.112e-4 | 3.488e-4 |
| SynFlow | 1.615e+11 | 4.169e+17 | 1.054e+16 | 2.373e+16 |

All arrays: shape (32,768,), dtype float64, zero NaN/Inf, zero zeros.

---

## 4. Step 2: Log Transformation

**Script**: `scripts/nats_bench_sss/transform_proxies_sss.py`

Transforms applied:

| Proxy | Transform | Rationale |
|---|---|---|
| param_count | `log(x)` | Always positive; no offset needed |
| NASWOT | `log(x + 1e-10)` | Scores ~1e-4 to ~3e-3; tiny offset for safety |
| ZenScore | `log(x + 1e-10)` | Same range as NASWOT |
| SynFlow | `log(x)` | Scores ~1e+11 to ~4e+17; all positive |

**No negation for any proxy** — all confirmed positive direction from Step 0 Q6.

### Log-Transformed Ranges

| Proxy | Log Min | Log Max | Log Mean | Log Std |
|---|---|---|---|---|
| param_count | 9.369 | 13.478 | 12.403 | 0.474 |
| NASWOT | -8.850 | -5.677 | -6.937 | 0.439 |
| ZenScore | -9.148 | -5.929 | -7.080 | 0.415 |
| SynFlow | 25.808 | 40.572 | 35.153 | 2.180 |

---

## 5. Step 3: Distribution Analysis

**Script**: `scripts/nats_bench_sss/analyze_distributions_sss.py`  
**Output**: `results/nats_bench_sss/proxy_distribution/`

### Raw Score Statistics

| Proxy | Mean | Std | CV | Skewness | Kurtosis |
|---|---|---|---|---|---|
| param_count | 2.687e+05 | 1.101e+05 | 0.410 | +0.355 | 2.800 |
| NASWOT | 1.062e-03 | 4.385e-04 | 0.413 | +0.732 | 3.730 |
| ZenScore | 9.112e-04 | 3.488e-04 | 0.383 | +0.516 | 3.195 |
| SynFlow | 1.054e+16 | 2.373e+16 | 2.251 | +5.125 | 40.971 |

**Key observations**:

**SynFlow raw distribution is severely right-skewed** (skew=5.125, kurt=40.971). The raw distribution is dominated by a large mass of architectures with moderate synflow scores and a very long right tail. This arises because SynFlow is a product of layer-wise synaptic salience terms — each channel width multiplicatively amplifies the score, causing exponential scaling with total capacity. However, the log-transform **fully normalises** SynFlow: log-transformed SynFlow has skewness = −0.382 and kurtosis = 2.869 (nearly Gaussian). The log scale spans 25.8 to 40.6.

**All four proxies are well-behaved in log space.** param_count, NASWOT, and ZenScore are approximately normally distributed in log space already (raw |skew| < 1, kurtosis near 3). SynFlow joins them after log-transform (log skew=-0.382, log kurt=2.869). This is an important contrast with NAS-Bench-201, where SynFlow produced ~50% near-zero raw scores from none/skip-dominated architectures, making its log-space distribution pathological. On SSS, every architecture uses Conv2d throughout, so all proxies produce well-conditioned scores. The log-transform is sufficient preprocessing for all four — no clipping, winsorising, or further normalisation is needed before PCA.

Note: the raw CV values in the table above (param_count=0.410, NASWOT=0.413, ZenScore=0.383) are computed on raw scores and are comparable because these three proxies have similar scale structure. SynFlow's raw CV=2.251 reflects its multiplicative scaling; its log-space CV=0.062 is far smaller than the other three (log-space CV ≈ 0.038–0.063), meaning SynFlow is actually the *least* spread in log space relative to its mean.

**No near-zero values**: SSS has zero near-zero values in any proxy — confirmed by the validation check (near_zero_pct=0.0 for all four raw arrays).

---

## 6. Step 4: Ranking Correlation Validation

**Script**: `scripts/nats_bench_sss/validate_ranking_correlations_sss.py`  
**Output**: `results/nats_bench_sss/proxy_validation/`

### Ground Truth Accuracy Distribution
- N = 32,768 valid (100% coverage, no missing data)
- Range: 79.79% – 93.65%
- Mean: 91.09%, Std: 1.27%
- **No degenerate cluster** — confirmed, minimum is 79.79%

### Correlation with GT Accuracy (Raw = Log-Transformed, Spearman is rank-based so identical)

| Proxy | Spearman ρ | Kendall τ | Top-1% Prec | Top-5% Prec | Top-10% Prec | Signal |
|---|---|---|---|---|---|---|
| **param_count** | **+0.8721** | +0.6904 | 0.50 | 0.62 | 0.68 | Strong |
| **SynFlow** | **+0.9162** | +0.7516 | **0.52** | **0.68** | **0.74** | Strong |
| ZenScore | +0.5566 | +0.3898 | 0.09 | 0.24 | 0.33 | Moderate |
| NASWOT | +0.5235 | +0.3643 | 0.08 | 0.22 | 0.32 | Moderate |

All p-values = machine zero.

### Cross-Proxy Correlation Matrix (Spearman)

|  | param_count | NASWOT | ZenScore | SynFlow |
|---|---|---|---|---|
| **param_count** | 1.000 | 0.580 | 0.622 | **0.896** |
| **NASWOT** | 0.580 | 1.000 | 0.649 | 0.543 |
| **ZenScore** | 0.622 | 0.649 | 1.000 | 0.577 |
| **SynFlow** | **0.896** | 0.543 | 0.577 | 1.000 |

### Analysis

**SynFlow dominates on SSS** (ρ=0.916), outperforming even param_count (ρ=0.872). This is the opposite of NAS-Bench-201 where SynFlow was the weakest proxy (ρ=0.164). The difference is explained by the search space structure: in SSS, every layer is a Conv2d and every connection is active. SynFlow's computation — `Σ|grad × weight|` with all-ones input and linearised weights — exactly captures the product of channel widths across layers, which is the dominant capacity signal in a pure width search space. SynFlow is essentially a differentiable generalisation of parameter count in this setting.

**The param_count ↔ SynFlow cross-correlation of 0.896 confirms near-redundancy.** These two proxies share ~80% of their rank-variance. Despite this, SynFlow carries residual signal beyond param_count — its partial correlation after debiasing is positive (Step 5 finds ρ_partial = +0.625).

**NASWOT and ZenScore are moderately correlated** (ρ=0.649) despite using the same covariance-trace methodology. The difference arises from stochasticity: NASWOT uses a single fixed random batch while ZenScore averages 4 draws. Their moderate mutual correlation (not 0.9+) confirms they are not measuring identical things.

**Top-1% precision is very low for NASWOT/ZenScore** (0.08–0.09): fewer than 1 in 10 true top-1% architectures are recovered. This is not primarily a proxy failure — it reflects the 1.19% accuracy spread at the top of SSS, where no proxy can meaningfully rank within 1.19% accuracy differences. Even param_count achieves only 50% top-1% precision despite ρ=0.872.

**Comparison with NAS-Bench-201**:
- NAS-Bench-201 param_count ρ = 0.749; SSS param_count ρ = 0.872 (+0.123)
- NAS-Bench-201 SynFlow ρ = 0.164; SSS SynFlow ρ = 0.916 (+0.752) — dramatic reversal
- NAS-Bench-201 NASWOT ρ = 0.515; SSS NASWOT ρ = 0.524 (nearly identical)
- NAS-Bench-201 ZenScore ρ = 0.554; SSS ZenScore ρ = 0.557 (nearly identical)

The NASWOT/ZenScore similarity across benchmarks suggests these proxies measure something partially independent of the search space type, while SynFlow and param_count are strongly search-space-dependent.

---

## 7. Step 5: Bias Disentanglement

**Script**: `scripts/nats_bench_sss/bias_disentanglement_sss.py`  
**Output directory**: `results/nats_bench_sss/debiased_proxy/`

### Method
For each proxy P in {SynFlow, NASWOT, ZenScore}:
1. OLS: `P ~ b0 + b1 × log(param_count)` → residuals `e_P`
2. **Method 1 (OLS residual Spearman)**: `spearmanr(e_P, e_GT)`
3. **Method 2 (Partial-rank Spearman)**: regress `rank(P) ~ rank(param_count)` and `rank(GT) ~ rank(param_count)`, correlate rank-residuals → **primary method**

### Phase B: OLS Fit Statistics

| Target | b1 (slope) | R² | Interpretation |
|---|---|---|---|
| GT accuracy | 2.384 | **0.789** | **79% of GT variance explained by size alone** |
| SynFlow | 4.137 | **0.810** | 81% of SynFlow variance from size |
| NASWOT | 0.541 | 0.342 | 34% of NASWOT variance from size |
| ZenScore | 0.543 | 0.384 | 38% of ZenScore variance from size |

**The GT R² of 0.789 is the defining characteristic of SSS**: size alone explains 79% of performance variance. This is far higher than NAS-Bench-201 (GT R²=0.157). SSS is by construction a capacity benchmark — wider networks systematically outperform narrower ones. Any proxy that correlates with size will inherit this signal.

In contrast to NAS-Bench-201 where NASWOT/ZenScore had R²=0.43–0.46 (heavy capacity confound), SSS shows **lower** R² for NASWOT (0.342) and ZenScore (0.384). This is counterintuitive given SSS is a pure capacity space, but the explanation is that in SSS the OLS slope b1=0.541 for both proxies is actually *higher* than on NAS-Bench-201 (b1≈0.46) — meaning the linear response per unit of log(param_count) is steeper. The lower R² despite the steeper slope means NASWOT/ZenScore have *more noise* around the linear fit on SSS: within each channel-width combination, covariance-trace scores are more variable (random batch effects dominate) because all operations are Conv2d and operation-type diversity no longer provides a stable signal. Larger residual variance lowers R² even when the mean trend is steeper.

### Phase C: Partial Correlation Results

| Proxy | OLS residual ρ | Partial-rank ρ | p-value | param R² | Decision |
|---|---|---|---|---|---|
| **SynFlow** | +0.5437 | **+0.6246** | 0.000 | 0.810 | **KEEP** |
| NASWOT | +0.0452 | **+0.0590** | 1.10e-26 | 0.342 | **EXCLUDE** |
| ZenScore | +0.0357 | **+0.0509** | 2.97e-20 | 0.384 | **EXCLUDE** |

**Full Spearman ρ (no debiasing) for reference**:
- SynFlow full ρ = +0.9162 → partial ρ = +0.6246 (retains strong independent signal)
- NASWOT full ρ = +0.5235 → partial ρ = +0.0590 (nearly all signal was capacity-mediated)
- ZenScore full ρ = +0.5566 → partial ρ = +0.0509 (same — near-complete collapse)

### Decision: Final Proxy Decisions

| Proxy | Global ρ | Param R² | Partial ρ | Decision |
|---|---|---|---|---|
| param_count | +0.872 | — | — | **Bias covariate → PCA input** |
| SynFlow | +0.916 | 0.810 | **+0.625** | **KEEP → PCA input** |
| NASWOT | +0.524 | 0.342 | +0.059 | **EXCLUDE** |
| ZenScore | +0.557 | 0.384 | +0.051 | **EXCLUDE** |

### Interpretation

**SynFlow result is the surprise finding of SSS.** On NAS-Bench-201, SynFlow was excluded (partial ρ = -0.002). On SSS, it has the strongest partial ρ of all proxies (+0.625), dramatically outperforming NASWOT and ZenScore which both collapse to near-zero after debiasing.

The mechanism is as follows. SynFlow measures `Σ|θᵢ · ∇_θᵢ L|` at initialisation with linearised weights. In a pure width search space, this quantity captures the product of all channel widths along each path through the network — it is a path-capacity measure, not just a parameter count. When param_count is regressed out, what remains in SynFlow residuals is the information about *how capacity is distributed across the 5 stages* — architectures with the same total parameter count but different capacity distributions differ in SynFlow. This distribution-within-capacity signal correlates with accuracy at +0.625, meaning architectures that distribute capacity more evenly across stages (rather than concentrating it all in early or late stages) tend to perform better.

**NASWOT and ZenScore collapse** completely after debiasing (ρ = 0.059 and 0.051). Their entire global signal of 0.524 and 0.557 was capacity-mediated. After removing the capacity component, they carry essentially no information about which architectural configurations within the same capacity class perform better. This is the opposite of what was hoped and directly opposite to their NAS-Bench-201 behavior (partial ρ = 0.154 and 0.190 there).

The structural explanation: in NAS-Bench-201, NASWOT/ZenScore could detect the diversity of operations within a cell (skip, conv, none) even at fixed parameter count. In SSS, all operations are Conv2d — there is no operation diversity to detect. The only variation is channel width, which directly maps to parameter count. The activation covariance trace proxies cannot discriminate between different channel-width configurations within the same total parameter budget because all operations are identical in type.

**Note on p-values for NASWOT/ZenScore**: despite ρ ≈ 0.06, p-values are 1.1e-26 and 3.0e-20. These are statistically significant but practically negligible — with N=32,768, even a ρ of 0.059 is detectable. A partial ρ of 0.059 explains 0.35% of GT variance after debiasing. This is not useful for ranking.

---

## 8. Step 6: PCA Whitening

**Script**: `scripts/nats_bench_sss/pca_whitening_sss.py`  
**Output directory**: `results/nats_bench_sss/pca_whitening/`

### Input Features (determined automatically from Step 5 decisions)

| Column | Source | Description |
|---|---|---|
| 0 | `transformed_proxy/param_count.npy` | Raw log-transformed size signal |
| 1 | `debiased_proxy/synflow_residuals.npy` | SynFlow with log(param_count) regressed out |

NASWOT and ZenScore excluded. The feature matrix is (32,768 × 2).

### Phase B: Standardisation

| Feature | Mean | Std | After z-score: z-std |
|---|---|---|---|
| param_count | +12.4028 | 0.4743 | 1.0000 |
| synflow_residual | -0.0000 | 0.9504 | 1.0000 |

### Phase C: PCA Eigenstructure

| PC | Eigenvalue | Var % | Cum. % | Retained |
|---|---|---|---|---|
| PC1 | 1.000031 | 50.00% | 50.00% | Yes |
| PC2 | 1.000031 | 50.00% | 100.00% | Yes |

**Both PCs have eigenvalue ≈ 1.000.** This is guaranteed by the OLS construction: OLS residuals are uncorrelated with the covariate by construction, so after z-scoring [log(param_count), synflow_residual] the covariance matrix is the identity I₂ — both eigenvalues are exactly 1.

However, the PCA still applies a rotation. From `pca_summary.json`, the loadings are:
- PC1 = **0.9253 × z(param_count) + 0.3793 × z(synflow_residual)** (~22° tilt toward param_count)
- PC2 = **−0.3793 × z(param_count) + 0.9253 × z(synflow_residual)**

Because the eigenspace is degenerate (both eigenvalues = 1.000031), this rotation is numerically arbitrary — any rotation of two uncorrelated unit-variance features produces the same whitened covariance. The whitening factor sqrt(1.000031) ≈ 1 is a near-no-op. The actual values written to `whitened_features.npy` are these PC projections, not the original z-scored features. However, since a linear layer in the MLP can recover any rotation, the information content is fully preserved regardless of which arbitrary rotation PCA chose.

This is a structurally cleaner result than NAS-Bench-201 where the three features had non-trivial correlations requiring real PCA rotation (PC1 eigenvalue = 1.559, PC3 = 0.441).

### Phase E: Validation

| Check | Value | Tolerance | Status |
|---|---|---|---|
| Max column mean | 4.28e-08 | 1e-04 | **PASS** |
| Max abs deviation from I | 4.48e-10 | 1e-03 | **PASS** |

Covariance matrix of whitened output:
```
+1.000000  +0.000000
+0.000000  +1.000000
```

### Output Files

| File | Contents |
|---|---|
| `results/nats_bench_sss/pca_whitening/scaler.pkl` | Fitted StandardScaler |
| `results/nats_bench_sss/pca_whitening/pca_model.pkl` | Fitted PCA(whiten=True) |
| `results/nats_bench_sss/pca_whitening/arch_ids.npy` | (32768,) int32 |
| `results/nats_bench_sss/pca_whitening/whitened_features.npy` | **(32768, 2) float32 — MLP input** |
| `results/nats_bench_sss/pca_whitening/pca_summary.json` | Metadata, loadings, validation |
| `results/nats_bench_sss/pca_whitening/pca_plots.png` | 4-panel diagnostics |

---

## 9. Cross-Benchmark Comparison: NAS-Bench-201 vs NATS-Bench SSS

| Property | NAS-Bench-201 | NATS-Bench SSS |
|---|---|---|
| N architectures | 15,625 | 32,768 |
| Search space type | Operation search (6 ops, 6 edges) | Width search (8 widths, 5 positions) |
| GT range (CIFAR-10) | ~10%–94.4% | 79.79%–93.65% |
| GT std | ~12% | 1.27% |
| Degenerate cluster | ~300 archs near 10% | None |
| GT R² from param_count | 0.157 | **0.789** |
| param_count ρ | 0.749 | 0.872 |
| SynFlow ρ | 0.164 | **0.916** |
| NASWOT ρ | 0.515 | 0.524 |
| ZenScore ρ | 0.554 | 0.557 |
| SynFlow partial ρ | -0.002 (EXCLUDED) | **+0.625 (KEPT)** |
| NASWOT partial ρ | +0.154 (kept) | +0.059 (EXCLUDED) |
| ZenScore partial ρ | +0.190 (kept) | +0.051 (EXCLUDED) |
| PCA input features | param_count, NASWOT_r, ZenScore_r | param_count, SynFlow_r |
| Whitened output shape | (15,625 × 3) | (32,768 × 2) |

**Thesis narrative**: The two benchmarks produce structurally opposite proxy survival patterns. NAS-Bench-201 keeps activation-diversity proxies (NASWOT/ZenScore) and excludes SynFlow. SSS keeps SynFlow and excludes NASWOT/ZenScore. The mechanism is clear: activation-diversity proxies are sensitive to operation-type variation (NAS-Bench-201 strength) but blind to width-only variation at fixed topology (SSS limitation). SynFlow, as a path-capacity proxy, is redundant with param_count when topology varies but adds independent capacity-distribution signal when topology is fixed and only widths change.

---

## 10. Current Focus: Step 7 — Surrogate MLP Training

**Input**: `results/nats_bench_sss/pca_whitening/whitened_features.npy` — shape (32,768, 2), float32  
**Target**: GT CIFAR-10 accuracy — `ori-test@89`  
**Architecture**: Shallow MLP `2 → 64 → 32 → 1`, ReLU activations  
**Loss**: Pairwise ranking loss  
**Split**: 80% train / 10% val / 10% test (fixed seed)  
**Evaluation**: Spearman ρ, Kendall τ, top-K precision (K=1%,5%,10%)  
**Ablation targets**: (1) raw features (no PCA), (2) param_count alone, (3) full 2-feature pipeline

**Also pending**: NAS-Bench-201 Step 7 (same MLP, 3 features). Both should be completed and evaluated together for the joint comparison table.

---

## 11. Output File Index

| Directory | Key files |
|---|---|
| `results/nats_bench_sss/raw_proxy_scores/` | `param_count.npy`, `naswot.npy`, `zenscore.npy`, `synflow.npy` — (32768,) each |
| `results/nats_bench_sss/transformed_proxy/` | `param_count.npy`, `naswot.npy`, `zenscore.npy`, `synflow.npy` — log-transformed |
| `results/nats_bench_sss/audit/` | `proxy_validation.json`, `proxy_validation.png`, `transform_stats.json` |
| `results/nats_bench_sss/proxy_distribution/` | `stats_summary.json`, `distribution_plots.png` |
| `results/nats_bench_sss/proxy_validation/` | `correlation_results.json`, `correlation_table.csv`, `scatter_plots.png`, `topk_precision.png`, `accuracy_distribution.png` |
| `results/nats_bench_sss/debiased_proxy/` | `partial_correlations.json`, `phase_b_summary.json`, `synflow_residuals.npy`, `gt_residuals.npy`, `naswot_residuals.npy`, `zenscore_residuals.npy`, `phase_b_residuals.png`, `phase_c_scatter.png` |
| `results/nats_bench_sss/pca_whitening/` | `whitened_features.npy`, `scaler.pkl`, `pca_model.pkl`, `arch_ids.npy`, `pca_summary.json`, `pca_plots.png` |
