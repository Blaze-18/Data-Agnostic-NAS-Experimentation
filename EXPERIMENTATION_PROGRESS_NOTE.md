# Zero-Cost Proxy Experimentation - Progress & Analysis Notes

**Date**: April 24, 2026  
**Phase**: Steps 1â€"6 Complete | Ready for Surrogate MLP Training (Step 7)  
**Status**: PCA whitening complete â€" whitened 3-feature matrix (15,625 Ã— 3) ready as MLP input; all 3 PCs retained; validation PASS

---

## 0. Proposed Framework Overview 

**Title**: Bias-Disentangled Structural Proxy Embedding Framework for Data-Agnostic Neural Architecture Ranking

**Core objective**: Estimate architecture performance using only structural zero-cost proxy signals â€” no dataset access during inference. The surrogate is trained offline on NAS benchmark ground-truth accuracies and then operates fully data-agnostically during search.

### Five-Stage Pipeline

| Stage | Name | Description |
|-------|------|-------------|
| **1** | Structural Proxy Extraction | Compute SynFlow, Zen-Score, NASWOT, Param Count at initialization via single forward/structural pass. No data, labels, or real gradients used. |
| **2** | Bias Disentanglement | Regress out structural size factors (param count, depth, width) from each proxy via OLS. Residual `ÎµÌƒáµ¢` is the bias-corrected signal. |
| **3** | Proxy Embedding Construction | Apply PCA + whitening to the debiased proxy matrix â†’ orthogonal embedding `Z = W(PÌƒ âˆ’ Î¼)`. Removes multicollinearity; standardises variance so param count cannot dominate. |
| **4** | Surrogate Training (Ranking Objective) | Shallow MLP trained on `Z` with a pairwise ranking loss `L_rank = Î£ max(0, âˆ’(yáµ¢âˆ’yâ±¼)(Å·áµ¢âˆ’Å·â±¼))` instead of MSE. Optimises for rank ordering, not point accuracy. |
| **5** | Data-Agnostic Inference | At search time: extract proxies â†’ apply precomputed bias correction â†’ project via stored PCA â†’ MLP outputs performance score. Zero data dependency. |

### Proxies Used

| Proxy | Role | Data-Agnostic |
|-------|------|---------------|
| SynFlow | Gradient flow magnitude; detects vanishing-gradient-prone architectures | âœ“ |
| Zen-Score | Activation expressivity under random perturbations | âœ“ |
| Param Count | Structural capacity (size/depth/width indicator); **bias covariate** | âœ“ |
| NASWOT | ReLU activation pattern diversity under random inputs | âœ“ |

### Validation Metrics (per thesis Â§5.6)
- **Primary**: Spearman Ï and Kendall Ï„ rank correlation coefficients
- **Downstream**: Top-10 selection accuracy (Pareto-optimal candidate retrieval)
- **Ablation**: Raw ensemble vs. linear surrogate vs. full framework (with/without bias calibration and whitening)

---

## 1. Pipeline Completion Status

| Step | Task | Status |
|------|------|--------|
| 1 | Proxy computation â€” all 4 proxies, 15,625 architectures | âœ… Complete |
| 2 | Log transformation with negation correction (NASWOT, Zen-Score) | âœ… Complete |
| 3 | Distribution analysis | âœ… Complete |
| 4 | Ranking correlation validation against CIFAR-10 ground truth | âœ… Complete |
| 5 | Bias disentanglement (regress out param_count) | âœ… Complete |
| 6 | PCA whitening -- corrected: param_count raw + NASWOT/ZenScore residuals | ✅ Complete |
| 7 | Surrogate MLP training + evaluation | âŒ Not started â€" NEXT |

**Implementation notes:**
- NASWOT and Zen-Score use multi-layer Conv2d hooking â€” all Conv2d layers are hooked and the mean of per-layer covariance traces is used as the score. This achieved 100% coverage vs. 26â€“31% with single-layer hooking in earlier experiments.
- SynFlow uses the Tanaka et al. 2020 algorithm: all-ones input, weight linearisation (`abs_()`), `loss = output.sum()`, score = `Î£|grad Ã— weight|`.
- NASWOT and Zen-Score raw scores are negated before log-transform (`-log(x + Îµ)`) because higher activation covariance trace correlates with skip-heavy (low accuracy) architectures â€” the raw correlation is negative.

---

## 2. Correlation Results (Transformed Proxies vs CIFAR-10 Test Accuracy, Epoch 199)

| Proxy | Spearman Ï | Kendall Ï„ | Top-1% Precision | Top-10% Precision | Signal |
|-------|-----------|-----------|-----------------|-------------------|--------|
| Param Count | **0.749** | 0.574 | 9.0% | 47.1% | Strong |
| Zen-Score | **0.554** | 0.390 | 3.2% | 25.4% | Moderate |
| NASWOT | **0.515** | 0.360 | 1.3% | 24.3% | Moderate |
| SynFlow | 0.164 | 0.114 | 10.3% | 41.4% | Weak |

All p-values are 0.0 (machine zero) except SynFlow (p = 4.2Ã—10â»â¹â´), confirming statistical significance across all proxies.

---

## 3. Key Findings and Analysis

### 3.1 Param Count is a Bias Covariate, Not a Predictor

Param Count achieves the strongest correlation (Ï = 0.749) but this is a structural bias: larger networks dominate the top of NAS-Bench-201 rankings purely because they have more capacity. The pairwise heatmap shows its cross-correlation with the other proxies:

- Param Count â†” NASWOT: **Ï = 0.59**
- Param Count â†” Zen-Score: **Ï = 0.62**

This means a substantial fraction of NASWOT's and Zen-Score's apparent correlation with accuracy is actually inherited from their shared correlation with network size. Param Count is designated a **bias covariate** â€” it will be regressed out in Step 5, not used as a predictor in the surrogate model.

The raw and transformed distributions of Param Count show a **discrete multi-modal structure** (3â€“4 peaks in the KDE). This is not a smooth continuous variable â€” NAS-Bench-201 architectures fall into discrete capacity classes. Standard OLS regression treats it as continuous, which is an approximation. Step 5 will compute both OLS residuals and partial Spearman Ï; if they disagree, partial Spearman Ï is the credible result.

### 3.2 NASWOT and Zen-Score: Redundancy Undecided

The expected prediction was that NASWOT â†” Zen-Score inter-proxy Ï would exceed 0.90, since both measure `trace(Cov(activations))` across Conv2d layers. The pairwise heatmap shows **Ï = 0.62** â€” moderate, not near-identical. The likely explanation is Zen-Score's multi-sample averaging (4 random input draws) vs NASWOT's fixed single batch; the stochasticity decorrelates scores at the architecture level even when marginal distributions look similar.

However, Ï = 0.62 in a small feature set is still substantial shared variance. **Neither proxy is excluded at this stage.** The Step 5 partial Ï measurements â€” after param_count is regressed out â€” are required before any inclusion decision is made. If one proxy's partial Ï collapses and the other's does not, the weaker one is dropped. If both partial Ï values remain above the threshold, both are retained as predictors with documented evidence of partial independence.

### 3.3 SynFlow: Weak Signal with Degenerate Distribution

SynFlow has Ï = 0.164 globally. The top-1% precision of 10.3% is 10Ã— better than random, but this is a coincidental property of its heavy right tail, not a reliable signal â€” the rank-rank scatter shows near-uniform horizontal noise across the full rank range.

The transformed distribution is degenerate: over 50% of architectures cluster at log(SynFlow) â‰ˆ 0â€“0.3 (these are skip/none-dominated architectures where SynFlow â‰ˆ 1.0, and log(1.0) = 0). The remaining scores form a disconnected tail reaching to 61. Q25=0.14, median=0.30, mean=6.9, max=61.3.

SynFlow is being carried through bias disentanglement solely to produce a formal measurement for the thesis ablation. The expectation â€” based on Ï=0.164 and the degenerate distribution â€” is exclusion. It is not a viable candidate for the surrogate predictor.

Separately: NASWOT and Zen-Score fail at top-1% precision (1.3% and 3.2%) not because of a proxy failure but because the top 10% of NAS-Bench-201 architectures compress into a 1.5% accuracy window (92.9â€“94.4%). No proxy can meaningfully rank within 1.5% accuracy differences. This is a benchmark ceiling characteristic.

### 3.5 Degenerate Architecture Cluster

The accuracy distribution shows ~300 architectures near 10% accuracy. These are skip/none-dominated cells that fail to learn entirely. Every proxy correctly pushes these to the bottom of its ranking, which inflates all Ï values relative to what they would be on the competitive subset alone (75â€“94%). Bias disentanglement will inherit this inflation; the partial Ï values post-debiasing are the more honest signal strength estimates.

---

## 4. Pairwise Inter-Proxy Correlation Summary (Heatmap)

|  | SynFlow | NASWOT | Zen-Score | Param Count | GT Accuracy |
|--|---------|--------|-----------|-------------|-------------|
| **SynFlow** | 1.00 | 0.13 | 0.13 | 0.23 | 0.16 |
| **NASWOT** | 0.13 | 1.00 | 0.62 | 0.59 | 0.52 |
| **Zen-Score** | 0.13 | 0.62 | 1.00 | 0.62 | 0.55 |
| **Param Count** | 0.23 | 0.59 | 0.62 | 1.00 | 0.75 |
| **GT Accuracy** | 0.16 | 0.52 | 0.55 | 0.75 | 1.00 |

**SynFlow is near-orthogonal to all other proxies** (Ï â‰¤ 0.23). This is the only positive property it retains â€” if it survives bias disentanglement, it contributes a genuinely independent axis of information.

---


## 5. Step 5 Results: Bias Disentanglement

**Script**: `scripts/nasbench201/bias_disentanglement.py`  
**Output directory**: `results/nasbench201/debiased_proxy/`

### 5.1 Method

For each proxy P in {SynFlow, NASWOT, Zen-Score} and for GT accuracy:
1. OLS regression: `P ~ b0 + b1 * param_count_transformed` -> save residuals `e_P`
2. **Method 1 (OLS residual Spearman):** `spearmanr(e_P, e_GT)`
3. **Method 2 (Partial rank Spearman):** regress `rank(P) ~ rank(param_count)` and `rank(GT) ~ rank(param_count)`, correlate rank-residuals
4. If `|Method1 - Method2| < 0.05` -> methods agree, use OLS. Otherwise use partial rank Spearman as primary (robust to discrete covariate).

### 5.2 OLS Fit Statistics (Phase B)

| Target | b1 (slope) | R2 | Interpretation |
|--------|------------|-----|----------------|
| GT accuracy | 4.84 | 0.157 | 15.7% of GT variance explained by size alone |
| SynFlow | 3.63 | 0.100 | 10.0% of SynFlow variance from size |
| NASWOT | 0.46 | 0.432 | **43.2%** of NASWOT variance from size |
| Zen-Score | 0.45 | 0.461 | **46.1%** of Zen-Score variance from size |

Nearly half the variance in NASWOT and Zen-Score is structural capacity, not activation quality signal.

### 5.3 Partial Correlation Results (Phase C)

| Proxy | OLS residual rho | Partial rank rho | Methods agree | Primary method | Partial rho |
|-------|-----------------|-----------------|---------------|----------------|------------|
| SynFlow | +0.1155 | **-0.0019** | No | partial_rank_spearman | **-0.002** |
| NASWOT | +0.3050 | **+0.1541** | No | partial_rank_spearman | **+0.154** |
| Zen-Score | +0.3377 | **+0.1895** | No | partial_rank_spearman | **+0.190** |

All three proxy pairs failed the method agreement test (|delta rho| > 0.05). The OLS residual method is optimistic in all cases because param_count is discrete (~28 unique values, 3-4 peaks) -- OLS residuals retain non-linear structure that the partial rank method correctly handles. Partial rank Spearman is the credible primary result for all proxies.

### 5.4 Decisions (Phase D)

| Proxy | Partial rho | p-value | Decision |
|-------|------------|---------|----------|
| SynFlow | -0.002 | 0.81 | **EXCLUDE** -- zero independent signal |
| NASWOT | +0.154 | 1.1e-83 | **KEEP (documented)** -- partial signal below 0.30 threshold |
| Zen-Score | +0.190 | 2.7e-126 | **KEEP (documented)** -- partial signal below 0.30 threshold |

**Decision threshold applied:** rho >= 0.30 -> KEEP; 0.10-0.30 -> KEEP_DOCUMENTED; < 0.10 -> EXCLUDE.

### 5.5 Interpretation

**SynFlow** is confirmed excluded. Its entire global rho of 0.164 was driven by correlation with architectural size. After removing that effect, rho = -0.002 (p = 0.81) -- indistinguishable from noise. It adds no independent information beyond param_count and would introduce noise into the surrogate input.

**NASWOT and Zen-Score** retain real but modest independent signal (rho = 0.15-0.19). Both p-values are effectively zero across 15,625 architectures, so the signal is statistically undeniable. However, both fall in the KEEP_DOCUMENTED band -- not the clean >=0.30 threshold. The OLS method gave rho = 0.30-0.34 but this was an artefact of param_count's discrete distribution inflating the OLS residual correlation.

**The key structural insight:** NAS-Bench-201's fixed macro skeleton (5 cells, 16 channels, constant depth and width) means param_count is the only varying structural covariate. Proxies that are theoretically sensitive to architectural expressivity (NASWOT, Zen-Score) end up heavily confounded with capacity because there is no depth or width variation to separate them. This is a known limitation of NAS-Bench-201 as a proxy evaluation benchmark.

**Surrogate MLP input features (post-Step 5):** `[param_count_transformed, naswot_residuals, zenscore_residuals]` -- SynFlow excluded. NASWOT and Zen-Score enter via their OLS residuals (param_count regressed out), not their raw transformed scores.

### 5.6 Output Files

| File | Contents |
|------|----------|
| `results/nasbench201/debiased_proxy/phase_a_verification.json` | Data alignment and NaN check report |
| `results/nasbench201/debiased_proxy/phase_b_summary.json` | OLS fit statistics (b0, b1, R2) per proxy |
| `results/nasbench201/debiased_proxy/phase_b_residuals.png` | Residual scatter vs param_count (4 panels) |
| `results/nasbench201/debiased_proxy/phase_c_scatter.png` | Residual vs GT-residual scatter (2x3 grid, both methods) |
| `results/nasbench201/debiased_proxy/partial_correlations.json` | Full results: both rho values, method agreement, decision per proxy |
| `results/nasbench201/debiased_proxy/{proxy}_residuals.npy` | OLS residuals for NASWOT, Zen-Score, SynFlow, GT |

---

## 6. Step 6 Results: PCA Whitening

**Script**: `scripts/nasbench201/pca_whitening.py`  
**Output directory**: `results/nasbench201/pca_whitening/`

### 6.1 Method

Pipeline applied to the corrected 3-feature matrix (15,625 x 3):

| Column | Source | Description |
|--------|--------|-------------|
| 0 | `param_count_transformed.json` | Raw log-transformed size signal — the bias covariate |
| 1 | `naswot_residuals.npy` | NASWOT with param_count regressed out (OLS, Step 5) |
| 2 | `zenscore_residuals.npy` | ZenScore with param_count regressed out (OLS, Step 5) |

NASWOT and ZenScore enter as their **debiased residuals**, not the raw transformed scores. This is the arrangement consistent with Step 5: param_count is the explicit size predictor; NASWOT/ZenScore contribute only their param_count-independent signal.

Steps:
1. **Standardise** -- z-score each feature (param_count std=1.059, naswot_residual std=0.555, zenscore_residual std=0.517)
2. **Fit PCA** -- eigen-decompose the 3x3 covariance matrix; record all eigenvalues and explained variance ratios
3. **Retain PCs** -- keep smallest k such that cumulative variance >= 99% (hard minimum k=2)
4. **Whiten** -- divide each PC score by sqrt(eigenvalue) --> output has exactly zero mean and identity covariance
5. **Validate** -- assert `|col_mean| < 1e-5` and `max|Cov(Z) - I| < 1e-4`

### 6.2 Input Feature Correlations (Pre-PCA)

| Feature pair | Pearson r | Note |
|---|---|---|
| param_count <-> naswot_residual | **~0.000** | Forced to zero by OLS construction |
| param_count <-> zenscore_residual | **~0.000** | Forced to zero by OLS construction |
| naswot_residual <-> zenscore_residual | 0.559 | Residual shared activation signal |

For reference, raw transformed score correlations were: param↔naswot=0.657, param↔zenscore=0.679, naswot↔zenscore=0.756. Residualisation reduced the naswot↔zenscore correlation from 0.756 to 0.559, confirming the debiasing removed the param_count-mediated component of their shared variance.

The residual-residual correlation of 0.559 motivates PCA: the two debiased proxies still share ~31% of variance (r²=0.312), so whitening ensures neither dominates the MLP input.

### 6.3 PCA Results

| PC | Eigenvalue | Var % | Cumulative % | Retained |
|---|---|---|---|---|
| PC1 | 1.559 | **52.0%** | 52.0% | Yes |
| PC2 | 1.000 | **33.3%** | 85.3% | Yes |
| PC3 | 0.441 | **14.7%** | 100.0% | Yes |

All 3 components retained. The eigenvalue structure is dramatically more balanced than the old run (52/33/15 vs 80/12/8), which is the expected result after residualisation: param_count no longer contributes its full raw variance to PC1 since it is now orthogonal to the residual columns by construction.

**PC2 eigenvalue = 1.000 exactly** (to 3 d.p.) confirms it is a purely residual axis — this is the unit-variance dimension that OLS guarantees for the param_count predictor itself after centering.

### 6.4 Validation Results

| Check | Result | Tolerance | Status |
|---|---|---|---|
| Max column mean | 2.42e-08 | 1e-05 | **PASS** |
| Max abs deviation from I | 1.03e-08 | 1e-04 | **PASS** |

Covariance matrix of whitened output:
```
+1.000000  +0.000000  -0.000000
+0.000000  +1.000000  +0.000000
-0.000000  +0.000000  +1.000000
```

### 6.5 Analysis and Interpretation

**The corrected eigenvalue structure is substantially healthier.** PC1 at 52% (not 80%) means the MLP input is far less dominated by a single direction. The three retained components now have a meaningful spread of information content.

**PC1 (52%) is the joint activation quality axis**, with loadings [0, 0.707, 0.707] — the equal-weight sum of naswot_residual and zenscore_residual. param_count loading is numerically zero (5×10⁻¹⁴). This is the consensus signal both debiased proxies share, and it is the largest single source of variance precisely because the two residuals correlate at r = 0.559 — their shared component is larger than either individual residual variance.

**PC2 (33%) is the pure param_count axis**, with loadings [1.000, 0, 0]. The eigenvalue of exactly 1.000 is guaranteed by the OLS construction: param_count is orthogonal to both residuals, so it forms its own independent eigenvector with eigenvalue equal to its standardised variance (= 1 by definition after z-scoring). This is the size channel.

**PC3 (15%) is the contrast axis**, encoding the difference between naswot_residual and zenscore_residual — the portion where NASWOT and ZenScore disagree after param_count is removed. This is the most genuinely independent piece of information in the feature set.

**The MLP now has balanced inputs.** No single PC dominates. The surrogate can in principle learn from the size signal (PC1), the shared activation quality signal (PC2), and the proxy-contrast signal (PC3) with equal representation after whitening.

**Limitation still applies:** The partial rho values from Step 5 (0.154 and 0.190) show that PC2 and PC3 together only have modest correlation with GT accuracy. The MLP cannot exceed what the input information allows. The ablation (raw features vs PCA-only vs full pipeline) will quantify how much whitening actually helps.

### 6.6 Output Files

| File | Size | Contents |
|------|------|----------|
| `results/nasbench201/pca_whitening/scaler.pkl` | 0.5 KB | Fitted `StandardScaler` (z-score params for inference) |
| `results/nasbench201/pca_whitening/pca_model.pkl` | 0.8 KB | Fitted `PCA(whiten=True)` (loadings + eigenvalues for inference) |
| `results/nasbench201/pca_whitening/arch_ids.npy` | 61 KB | `(15625,)` int32 -- row-to-arch_id alignment |
| `results/nasbench201/pca_whitening/whitened_features.npy` | 183 KB | `(15625, 3)` float32 -- MLP input features |
| `results/nasbench201/pca_whitening/pca_summary.json` | 1.4 KB | All metadata: variance ratios, loadings, scaler params, validation stats |
| `results/nasbench201/pca_whitening/pca_plots.png` | 238 KB | 4-panel diagnostic: input correlations, scree, PC1 vs PC2 scatter, output covariance |

---

## 7. Current Focus

**Step 6 is complete. The next task is surrogate MLP training (Step 7).**

### Final proxy and feature status

| Proxy | Global rho | Param R2 | Partial rho | Decision |
|-------|-----------|---------|------------|----------|
| Param Count | 0.749 | -- | -- | Bias covariate --> PCA input |
| Zen-Score | 0.554 | 0.461 | +0.190 | **KEEP** --> PCA input |
| NASWOT | 0.515 | 0.432 | +0.154 | **KEEP** --> PCA input |
| SynFlow | 0.164 | 0.100 | -0.002 | **EXCLUDED** |

| PCA Component | Eigenvalue | Var % | Role |
|---|---|---|---|
| PC1 | 1.559 | 52.0% | Joint activation quality axis (sum of naswot_r + zenscore_r) |
| PC2 | 1.000 | 33.3% | Pure param_count axis (orthogonal to residuals by OLS construction) |
| PC3 | 0.441 | 14.7% | Proxy contrast axis (naswot_r − zenscore_r difference signal) |

### Step 7: Surrogate MLP Training

**Input:** `results/nasbench201/pca_whitening/whitened_features.npy` -- shape `(15625, 3)`, float32  
**Target:** CIFAR-10 test accuracy at epoch 199 (`ori-test@199`) from `data/nasbench201/chunks_clean/arch2infos/`  
**Objective:** Learn a rank-preserving mapping from whitened proxy features to architecture performance score.

**Architecture:** Shallow MLP: `3 --> 64 --> 32 --> 1`, ReLU activations, scalar output

**Loss function:** Pairwise ranking loss (not MSE) -- optimises rank ordering directly, consistent with Spearman rho as evaluation metric:
```
L_rank = sum_{i,j} max(0, -(y_i - y_j)(y_hat_i - y_hat_j))
```

**Training protocol:**
- Split: 80% train / 10% val / 10% test (fixed seed for reproducibility)
- Optimiser: Adam
- Early stopping on validation ranking loss

**Evaluation metrics:** Spearman rho, Kendall tau, top-K precision (K = 1%, 5%, 10%) on held-out test set  
**Ablation targets:** (1) raw features (no PCA), (2) PCA without whitening, (3) full pipeline

**Output files (Step 7):**
- `results/nasbench201/surrogate_mlp/mlp_model.pkl` -- trained model
- `results/nasbench201/surrogate_mlp/training_curves.png` -- loss/metric vs epoch
- `results/nasbench201/surrogate_mlp/evaluation_results.json` -- all metrics on test set
- `results/nasbench201/surrogate_mlp/ablation_results.json` -- ablation comparison table
