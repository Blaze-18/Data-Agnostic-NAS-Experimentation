# Zero-Cost Proxy Experimentation - Progress & Analysis Notes

**Date**: April 21, 2026  
**Phase**: Steps 1â€“4 Complete | Ready for Bias Disentanglement (Step 5)  
**Status**: All proxies validated on full NAS-Bench-201 (15,625 architectures, CIFAR-10)

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
| 5 | Bias disentanglement (regress out param_count) | â³ Next |

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

## 5. Next Step: Bias Disentanglement (Step 5)

**Objective**: Isolate the portion of each proxy's correlation with accuracy that is independent of network size (param_count).

**Method**: For each proxy P âˆˆ {SynFlow, NASWOT, Zen-Score}:
1. OLS regression: `P_transformed ~ Î²â‚€ + Î²â‚ Ã— param_count_transformed` â†’ save residuals
2. Compute partial Spearman Ï: `spearmanr(P_residuals, GT_accuracy_residuals)` where GT residuals also have param_count regressed out
3. Compare OLS partial Ï vs direct partial Spearman Ï â€” if they diverge, report partial Spearman as primary (due to param_count's discrete distribution)

**Decision criteria post-debiasing:**
- Partial Ï â‰¥ 0.30 â†’ keep as surrogate predictor
- Partial Ï 0.10â€“0.30 â†’ keep with documentation of weakness
- Partial Ï < 0.10 â†’ exclude with documented evidence

**Expected outcomes:**
- NASWOT and Zen-Score: partial Ï will drop but likely remain above 0.30 (their cross-correlation with param_count is 0.59â€“0.62, not 0.90+)
- SynFlow: partial Ï likely drops below 0.10 (cross-correlation with param_count is only 0.23, but its global Ï is already only 0.164)
- Param Count: not evaluated â€” bias covariate only

**Output files (Step 5):**
- `results/debiased_proxy/{proxy}_debiased.json` â€” OLS residuals
- `results/debiased_proxy/partial_correlations.json` â€” partial Ï, p-values, decision per proxy

---

## 6. Current Focus

**The only task that matters right now is implementing and running bias disentanglement (Step 5).**

All proxy scores are computed and validated. No proxy inclusion decisions should be made before Step 5 results exist. The specific questions Step 5 must answer:

1. **NASWOT partial Ï** after regressing out param_count â€” does it stay above 0.30?
2. **Zen-Score partial Ï** after regressing out param_count â€” does it stay above 0.30? Is it higher or lower than NASWOT?
3. **SynFlow partial Ï** â€” formal measurement for ablation documentation. Expected: below 0.10 â†’ excluded.
4. **OLS vs partial Spearman agreement** â€” due to param_count's discrete multi-modal distribution, both must be computed. If they diverge, partial Spearman is the credible result.

**Proxy status going into Step 5:**

| Proxy | Global Ï | Status | Decision pending |
|-------|----------|--------|------------------|
| Param Count | 0.749 | Bias covariate only | Not evaluated as predictor |
| Zen-Score | 0.554 | Candidate | Partial Ï measurement required |
| NASWOT | 0.515 | Candidate | Partial Ï measurement required |
| SynFlow | 0.164 | Degenerate distribution | Expected exclusion â€” formal measurement for ablation |

**Note on multi-layer hooking:** The current NASWOT and Zen-Score scores (Ï=0.515, 0.554) were computed with multi-layer Conv2d hooking, which achieved 100% coverage vs 26â€“31% with single-layer. No A/B comparison of Ï values between the two implementations was done â€” only coverage was compared. The correlation numbers in Section 2 reflect the multi-layer implementation.

Do not proceed to PCA whitening, surrogate training, or proxy addition (GradNorm, topology metrics) until Step 5 partial Ï values are in hand.

---

*All content below this line is superseded and has been removed.*
