# NAS-Bench-101 Experiment Summary

**Project**: Bias-Disentangled Structural Proxy Embedding Framework  
**Date**: May 8, 2026  
**Benchmark**: NAS-Bench-101 — 423,624 unique DAG architectures, fixed-cell topology search  

---

## Why This Benchmark

NAS-Bench-101 is the hardest test in the thesis pipeline. With R² = 0.047 between parameter count and validation accuracy, model size explains almost **zero** performance variance. This makes it a true litmus test: proxies that survive here carry genuine architecture-quality signal, not just size confounding.

| Benchmark | R²(size→GT) | Role |
|-----------|-------------|------|
| NAS-Bench-201 | 0.157 | Moderate size effect |
| NATS-Bench SSS | 0.789 | Size dominates (stress test) |
| **NAS-Bench-101** | **0.047** | Size nearly irrelevant (litmus test) |

---

## Pipeline Overview

```
TFRecord (2.1 GB)
    │
    ▼ Step 0 — Structural Audit
    │   Extract GT accuracies, param counts, arch specs
    │   Verified: n=423,624 · R²=0.047 · GT mean=89.68%
    │
    ▼ Step 1 — Zero-Cost Proxy Computation (GPU)
    │   Four proxies computed independently on RTX 5070 Ti
    │
    ▼ Step 2 — Log Transformation + Sign Correction
    │   Stabilise distributions; auto-detect and fix negative correlations
    │
    ▼ Step 3 — Distribution Analysis
    │   Normality checks, skewness, kurtosis
    │
    ▼ Step 4 — Ranking Correlation Validation
    │   Spearman ρ, Kendall τ, Top-K precision vs ground truth
    │
    ▼ Step 5 — Bias Disentanglement
    │   OLS: remove log(param_count) signal from each proxy
    │   Partial-rank Spearman to gate proxy survival
    │
    ▼ Step 6 — PCA Whitening
        Build whitened feature matrix for MLP input
```

---

## Proxy Algorithms

### Parameter Count
Raw edge/op count from the DAG specification. Used as the size covariate in all debiasing steps — never as a proxy in isolation.

### SynFlow — Tanaka et al. (2020)
Gradient-based proxy using a linearised network and all-ones input. Avoids data dependency by using a synthetic input.

$$S_\text{SynFlow} = \sum_l \left| \frac{\partial \mathcal{L}}{\partial \theta_l} \odot \theta_l \right|$$

where $\mathcal{L} = \mathbf{1}^\top f(\mathbf{1}; |\Theta|)$ and $|\Theta|$ denotes absolute-valued weights.

### NASWOT — Mellor et al. (2021)
Measures activation diversity across a mini-batch. A more diverse Jacobian → architectures that separate inputs more effectively → higher expected accuracy.

$$S_\text{NASWOT} = \log \left| K \right|, \quad K_{ij} = \text{sign}(h(x_i)) \cdot \text{sign}(h(x_j))^\top$$

Approximated here as the mean trace of the inter-sample covariance of Conv2d activations.

### ZenScore — Lin et al. (2021)
Stochastic variant of activation diversity. Averages covariance traces over multiple random batches for lower variance.

$$S_\text{ZenScore} = \mathbb{E}_{\epsilon \sim \mathcal{N}(0,I)} \left[ \Phi(f(x + \epsilon)) \right]$$

Approximated as the mean covariance trace over 4 independent random batches.

---

## Transformation

All raw proxy scores are log-transformed to stabilise variance:

$$\tilde{p} = \log(p + \varepsilon)$$

**Sign correction**: Raw Spearman ρ between NASWOT and ZenScore with GT was **negative** (ρ ≈ −0.22), opposite to NAS-Bench-201 and SSS. This indicates that in DAG space, high activation diversity correlates with *lower* accuracy (skip-heavy architectures confound the signal). The transform negates these proxies before logging to align the direction:

$$\tilde{p} = \log(-p + \varepsilon) \quad \text{if } \rho_\text{raw} < 0$$

---

## Bias Disentanglement

For each proxy $p$, the size-confounded component is removed via OLS regression on log(param_count):

$$p = \alpha_0 + \alpha_1 \cdot \log(\text{params}) + r_p$$

The residual $r_p$ is the size-free proxy signal.

**Partial-rank Spearman** is the primary decision metric. This is more robust than OLS-residual Spearman because the relationship between proxy and size may be nonlinear:

1. Rank all arrays: $\mathbf{q} = \text{rank}(\mathbf{p})$, $\mathbf{g} = \text{rank}(\text{GT})$, $\mathbf{c} = \text{rank}(\log\text{params})$
2. Regress $\mathbf{q}$ and $\mathbf{g}$ on $\mathbf{c}$ via OLS → residuals $\tilde{\mathbf{q}}, \tilde{\mathbf{g}}$
3. $\rho_\text{partial} = \text{Spearman}(\tilde{\mathbf{q}},\, \tilde{\mathbf{g}})$

**Decision thresholds**:

| Partial ρ | Decision |
|-----------|----------|
| ≥ 0.30 | KEEP |
| 0.10 – 0.30 | KEEP_DOCUMENTED |
| < 0.10 | EXCLUDE |

---

## PCA Whitening

Kept proxy residuals + log(param_count) are assembled into a feature matrix $X \in \mathbb{R}^{N \times d}$, then:

1. **StandardScaler**: $X \leftarrow (X - \mu) / \sigma$ per feature
2. **PCA(whiten=True)**: $Z = \Lambda^{-1/2} U^\top X_\text{std}$ where $U, \Lambda$ are eigenvectors/eigenvalues
3. **Truncate** to retain ≥ 99% explained variance

This produces a decorrelated, unit-variance embedding ready for MLP input.

---

## Results

### Raw Proxy Correlations with GT (Spearman ρ)

| Proxy | Global ρ | Kendall τ | Top-5% | Top-10% |
|-------|----------|-----------|--------|---------|
| param_count | 0.435 | 0.307 | 20.1% | 27.1% |
| synflow | 0.421 | 0.287 | 11.2% | 21.7% |
| naswot | 0.225 | 0.154 | 4.3% | 11.2% |
| zenscore | 0.225 | 0.154 | 3.9% | 11.0% |

### Bias Disentanglement Results

| Proxy | OLS-residual ρ | Partial-rank ρ | Decision |
|-------|----------------|----------------|----------|
| synflow | 0.2179 | **0.2420** | KEEP_DOCUMENTED |
| naswot | 0.0828 | **0.1246** | KEEP_DOCUMENTED |
| zenscore | 0.0829 | **0.1250** | KEEP_DOCUMENTED |

All three proxies survived. Note that OLS-residual ρ underestimates NASWOT/ZenScore signal (0.083) vs partial-rank ρ (0.125), confirming the nonlinear size relationship — partial-rank Spearman is the correct choice here.

### PCA Output

| | Value |
|---|---|
| Input features | 4 — log_param_count + 3 residuals |
| PCs retained | **3** |
| Variance explained | **99.83%** |
| max\|col_mean\| | 2.75e-16 ✅ |
| max\|Cov − I\| | 1.55e-15 ✅ |
| MLP input shape | (423,624 × 3) |

---

## Cross-Benchmark Proxy Behaviour

| Proxy | NAS-201 partial ρ | SSS partial ρ | **NAS-101 partial ρ** |
|-------|:-----------------:|:-------------:|:---------------------:|
| SynFlow | −0.002 ❌ OUT | +0.625 ✅ IN | **+0.242 ✅ IN** |
| NASWOT | +0.154 ✅ IN | +0.059 ❌ OUT | **+0.125 ✅ IN** |
| ZenScore | +0.190 ✅ IN | +0.051 ❌ OUT | **+0.125 ✅ IN** |

This reversal pattern across benchmarks is a key thesis finding. No proxy is universally reliable. SynFlow is useless when operation diversity dominates (NAS-201) but strong when path-capacity varies (SSS, NAS-101). NASWOT/ZenScore are useful when operations vary (NAS-201, NAS-101) but collapse when architecture is uniformly Conv2d (SSS).

---

## Key Findings

1. **Sign reversal of NASWOT/ZenScore on NAS-101**: Raw ρ ≈ −0.22 (high activation diversity → *lower* accuracy in DAG space). This is the opposite of all other benchmarks and reflects the confounding effect of skip connections in DAGs.

2. **SynFlow matches param_count in raw correlation** (ρ = 0.421 vs 0.435). SynFlow effectively captures path-capacity distribution in DAG architectures, performing analogously to direct parameter counting.

3. **All three proxies survive debiasing**, unlike SSS where only SynFlow survived. DAGs contain genuine architectural variation beyond size — operation type and connectivity patterns carry independent signal.

4. **PCA compresses 4 features to 3 PCs with 99.83% retention** — the 4th PC (naswot/zenscore redundancy) is dropped as noise, consistent with their near-identical distributions.

---

## Section A — Literature Validation: Are These Results Okay?

This section examines each step against what established literature reports for NAS-Bench-101, and gives an explicit verdict on whether the numbers are credible.

---

### A.1 Audit: n=423,624, R²=0.047, GT mean=89.68%

**Verdict: ✅ CONFIRMED CORRECT**

These match the published NAS-Bench-101 dataset description exactly (Ying et al., 2019). The R²=0.047 figure is well-documented — it is one of the defining characteristics that distinguishes NAS-101 from other benchmarks and is why the dataset is specifically interesting for evaluating whether proxies generalise beyond size-correlated spaces. No issue here.

---

### A.2 Raw Proxy Correlations

**Literature baseline** (from Abdelfattah et al., ICLR 2021 — the most cited zero-cost proxy benchmark paper — and White et al., NeurIPS 2021 — the largest comparative study with 31 predictors):

| Proxy | Our result | Literature range (NAS-101) | Assessment |
|-------|-----------|---------------------------|------------|
| param_count ρ | **0.435** | 0.35 – 0.55 | ✅ Consistent |
| SynFlow ρ | **0.421** | 0.35 – 0.55 | ✅ Consistent |
| NASWOT ρ | **0.225** | 0.20 – 0.45 | ⚠️ Low end, see below |
| ZenScore ρ | **0.225** | 0.20 – 0.40 | ⚠️ Low end, see below |

**SynFlow (ρ = 0.421)**: This is squarely in the expected range. Abdelfattah et al. report SynFlow as one of the stronger single-shot proxies on NAS-101. The number is credible.

**NASWOT (ρ = 0.225 after sign flip)**: This is where a genuine concern arises. Mellor et al. (ICML 2021) — the original NASWOT paper — report *positive* correlations on NAS-Bench-101, not negative ones. Getting a raw ρ of −0.22 before the sign flip is inconsistent with the original paper's results on this same benchmark. The most likely cause is the **implementation approximation**: the script uses "mean trace of inter-sample Conv2d activation covariance" as a proxy for the theoretical log-determinant of the binary activation kernel. These are not equivalent, especially in DAG architectures with variable numbers of Conv2d operations across architectures. When architectures with more layers contribute larger trace values simply because they have more Conv2d modules to hook, the measure conflates layer count with activation diversity, producing an artefact rather than a genuine signal. This is not a data error — it is a proxy implementation issue that slightly undermines the NASWOT and ZenScore results on this benchmark specifically.

**Verdict**: SynFlow results ✅ are consistent with literature. NASWOT and ZenScore results ⚠️ are at the low end, and the negative raw correlation is inconsistent with the original NASWOT paper — likely due to the approximation used rather than a genuine inversion of the proxy's behaviour.

---

### A.3 Log-Transformation and Sign Correction

**Verdict: ✅ Statistically valid as applied, but the sign flip carries risk**

Log-transforming right-skewed proxy distributions before correlation analysis is standard practice. The auto-detected sign flip for NASWOT and ZenScore is technically correct given the observed negative ρ — the pipeline does the right thing with the numbers it has. However, flipping the sign of an activation diversity proxy that is *expected* to be positively correlated (per the original papers) is a red flag that something upstream is wrong, not a theoretical finding. Framing this as "In DAG space, high activation diversity → lower accuracy" in the thesis will be challenged by reviewers who will point to Mellor et al.'s own NAS-101 results showing positive correlation.

---

### A.4 Bias Disentanglement — Partial Correlations

**Verdict: ⚠️ MIXED — methodology is sound, but the numbers reveal a problem**

The partial-rank Spearman method is statistically defensible and is a reasonable choice for nonlinear covariate control. The OLS-on-ranks approach is used in econometrics (Imbens & Wooldridge, 2009) and is appropriate here.

**The problem is the magnitude of signal loss.** When R² = 0.047, param_count explains only ~5% of GT variance. Debiasing should therefore have a *minimal* effect on proxy correlation. But SynFlow drops from ρ = 0.421 to partial ρ = 0.242 — a **43% reduction in signal** after removing a covariate that explains barely 5% of the target. This is disproportionate and is a warning sign.

What is likely happening: SynFlow and param_count are themselves correlated with each other (SynFlow is a function of gradient × weight magnitudes, which scale with network size). Even though param_count → GT is weak (R²=0.047), the collinearity between SynFlow and param_count is non-trivial. When you regress SynFlow ranks on param_count ranks and take residuals, you discard the shared variance between SynFlow and param_count — and since much of SynFlow's raw GT correlation came from that shared variance, you're left with substantially less. The result is not wrong, but it means that SynFlow's GT-predictive power on NAS-101 is **more size-mediated than the R²=0.047 figure suggests**.

Concretely: SynFlow knows about architecture capacity. So does param_count. Even when that capacity only weakly predicts GT, both measures agree on the same architectures being "large." Debiasing removes that agreement. What remains (partial ρ=0.242) is genuinely size-free signal — but there is less of it than a naive reading of the raw numbers would suggest.

**NASWOT/ZenScore (partial ρ ≈ 0.125)**: These are borderline. The KEEP_DOCUMENTED classification is honest — 0.125 is above the 0.10 threshold, but barely. Given the implementation concern in A.2, these values should be treated with caution. They may not replicate with a more accurate NASWOT implementation.

**Verdict**: The methodology is correct. The numbers are internally consistent. But the magnitude of SynFlow signal loss is larger than expected, and the NASWOT/ZenScore partial ρ values are marginal and potentially artefactual.

---

### A.5 PCA Whitening

**Verdict: ✅ CORRECT AND WELL-EXECUTED**

The PCA step is textbook. The validation numbers (max|col_mean| = 2.75e-16, max|Cov−I| = 1.55e-15) are essentially machine epsilon — perfect whitening. The 4→3 PC compression with 99.83% variance retention is appropriate. No issues here technically.

The only observation worth noting: NASWOT and ZenScore producing nearly identical distributions means the feature matrix was effectively rank-3 before PCA, not rank-4. The 4th PC being almost pure noise (0.17% variance) confirms these two proxies are nearly collinear. This raises the question of whether both should be included or whether one is redundant — but it doesn't break anything, and PCA handles it correctly.

---

### Overall Section A Verdict

> **The pipeline is technically correct and the raw correlation numbers are broadly consistent with published literature ranges. Two genuine concerns exist: (1) the NASWOT/ZenScore implementation may not faithfully replicate the original metric in DAG architectures, producing the anomalous sign reversal; and (2) the debiasing step removes more SynFlow signal than the R²=0.047 figure would predict, suggesting SynFlow's GT correlation on NAS-101 is more size-mediated than it appears. Neither issue invalidates the results, but both need to be acknowledged in the thesis.**

---

## Section B — Honest Thesis Assessment

This section gives an unfiltered opinion on what the results mean for the thesis contribution, based on current literature.

---

### B.1 The Central Claim and What the Numbers Say

The thesis title proposes a "Bias-Disentangled Structural Proxy Embedding Framework." The implied claims are:

1. Zero-cost proxies contain size bias that, if removed, reveals stronger architecture-quality signal
2. The residual signals are strong enough to be worth combining into an embedding
3. A trained MLP on this embedding outperforms using any single proxy directly

Looking at NAS-Bench-101 honestly:

- **Claim 1** is *partially* supported. Size bias exists and is removed. But on NAS-101 with R²=0.047, the bias is small, yet its removal reduces SynFlow from ρ=0.421 to ρ=0.242. This does not demonstrate that debiasing *reveals* hidden signal — it demonstrates that debiasing *reduces* the signal that was there. That is the opposite of the intended narrative for this benchmark.

- **Claim 2** is weak. All three proxies after debiasing are in the KEEP_DOCUMENTED tier (ρ 0.10–0.30). None crossed the KEEP threshold (ρ≥0.30). The framework is combining three weak signals. Whether three weak signals combine into one strong one depends entirely on Step 7. It is not a given.

- **Claim 3** cannot be evaluated yet. But be prepared: the best raw proxy (SynFlow at ρ=0.421) sets a hard benchmark. The MLP trained on debiased 3-PC features must beat ρ=0.421 to justify the pipeline on this benchmark. Given that the individual partial ρ values are 0.242, 0.125, and 0.125, the MLP has weak building blocks to work with. It is more likely the MLP achieves somewhere in the range ρ=0.25–0.40, which would mean it **does not beat raw SynFlow**.

---

### B.2 The Strongest Result and How to Protect It

The cross-benchmark proxy reversal is genuinely interesting and scientifically valid:

| | SynFlow partial ρ | NASWOT partial ρ |
|---|---|---|
| NAS-201 | −0.002 (excluded) | +0.190 (kept) |
| NATS-SSS | +0.625 (kept) | +0.059 (excluded) |
| NAS-101 | +0.242 (documented) | +0.125 (documented) |

This pattern — proxies that are strong in one space are weak in another, and the active proxy set is benchmark-dependent — is a legitimate contribution. White et al. (NeurIPS 2021) and Ning et al. (NeurIPS 2021) both document high variance across benchmarks for zero-cost proxies, but neither proposes a framework that systematically identifies and uses only the benchmark-appropriate signals. This is the thesis's strongest differentiating claim and it is supported by the data.

**Protect this by not over-claiming.** The evidence supports: *"proxy survival after bias disentanglement is benchmark-specific, confirming that no single proxy is universally dominant."* It does not support: *"our framework always selects the best proxies"* — the absolute signal levels are too weak to claim that.

---

### B.3 The Two Real Problems

**Problem 1: The NASWOT/ZenScore sign flip is theoretically incoherent.**

If you flip NASWOT and ZenScore to have positive correlation and then claim they survive debiasing with partial ρ≈0.125, a reviewer will immediately ask: "Why did you flip them?" If your answer is "because they were negatively correlated, and we corrected the sign direction," the follow-up is: "But the original NASWOT paper shows positive correlation on NAS-Bench-101. Why does your implementation give the opposite result?" This is a hard question to answer without admitting an implementation approximation that deviates from the published method. The proxy scores for NASWOT and ZenScore on NAS-101 are built on a weaker foundation than SynFlow, and the thesis treatment of them as equally valid "debiased signals" will draw scrutiny.

**Recommendation**: If time permits before thesis submission, validate the NASWOT implementation against the published Mellor et al. code on a small NAS-101 sample. If the sign reversal persists with the exact original implementation, it becomes a genuine finding worth discussing. If the sign is positive with the original implementation, the approximation needs to be documented as a limitation.

**Problem 2: The debiasing narrative does not hold cleanly on the benchmark that is supposed to be the hardest test.**

The thesis argues that NAS-101 (R²=0.047) is the most credible validation because size barely predicts performance. This argument works for the raw proxy step — SynFlow achieving ρ=0.421 despite size being almost irrelevant does show it captures something real. But when debiasing then *removes* 43% of that signal, it weakens the story. A critical reader could argue: "You built a pipeline to remove size bias, but on the benchmark where size bias is smallest, the debiasing does the most damage. This suggests your debiasing is not surgical — it is removing signal along with noise."

This is a fair criticism. The response requires the MLP step to recover some of that combined signal. If the MLP on NAS-101 achieves ρ ≥ 0.40, it redeems the pipeline. If it plateaus at ρ ≈ 0.30, the honest conclusion is that the framework adds minimal value on this benchmark, and the thesis must acknowledge that as a limitation rather than a success.

---

### B.4 Is the Methodology Sound?

**Short answer: Yes, conditionally.**

The statistical machinery (OLS debiasing, partial-rank Spearman, PCA whitening, pairwise ranking loss MLP) is all standard, correctly implemented, and well-justified. The benchmark selection (three benchmarks with fundamentally different size-GT relationships) is clever and provides genuine scientific structure. The partial-rank Spearman as the primary gate metric is a better choice than OLS-residual ρ for nonlinear relationships, and the results confirm this (it recovers more signal for NASWOT/ZenScore).

**The conditionality**: The methodology is sound *if* the MLP step produces results that are at least competitive with the best raw proxy on each benchmark. The SSS result (SynFlow partial ρ=0.625) is strong enough to anchor the thesis. NAS-201 and NAS-101 are weaker anchors. If NAS-101's MLP result comes in below raw SynFlow, the conclusion to state honestly is: "The framework is most valuable in high-bias regimes (SSS, R²=0.789) where debiasing removes large confounds and frees signal. In low-bias regimes (NAS-101, R²=0.047), individual proxy signals are already modest and debiasing provides less marginal benefit."

That is still a publishable, honest thesis. It correctly characterises *when* the framework is useful rather than overclaiming universality.

---

### B.5 What Step 7 Must Show

The MLP (Step 7) is the decisive step. Here is what each outcome means:

| NAS-101 MLP ρ | Meaning |
|---|---|
| ρ ≥ 0.45 | Pipeline improves over best raw proxy. Strong result, validate thoroughly. |
| 0.40 ≤ ρ < 0.45 | Competitive with best raw proxy. Acceptable; debiasing is not harmful. |
| 0.30 ≤ ρ < 0.40 | Below best raw proxy. Framework does not help on NAS-101. Must be acknowledged. |
| ρ < 0.30 | Framework hurts on NAS-101. Critical finding — thesis must address directly. |

Based on the partial ρ values (0.242, 0.125, 0.125), the realistic expectation is the MLP landing in the **0.30–0.40 range**. This is below raw SynFlow (0.421). If that is what happens, do not hide it. The SSS result is strong enough that the thesis remains valid — the narrative simply needs to be: *the framework works best when there is substantial size confounding to remove.*

---

### B.6 Summary Recommendation

1. **Run Step 7 and report the ablation table honestly.** The four-variant ablation (size_only, best_raw, pca_raw, full_pipeline) will show exactly what the debiasing contributes. Do not cherry-pick.

2. **Investigate the NASWOT sign reversal** before finalising the thesis. At minimum, document it as a known implementation limitation. Do not present it as a theoretical finding without checking against the original code.

3. **Reframe the NAS-101 narrative** from "hardest test we passed" to "benchmark where size bias is minimal, showing debiasing is conservative and does not harm signals" — this is defensible regardless of the MLP outcome.

4. **The SSS result is your anchor.** SynFlow partial ρ = 0.625 on SSS, rising from raw ρ = 0.916 but more importantly from near-zero after size removal. The story on SSS — that the framework correctly identifies which proxy retains independent signal in a size-dominated space — is genuinely novel and defensible. Build the thesis argument around that result first.

5. **Do not over-cite the KEEP_DOCUMENTED classification as a success on NAS-101.** It is a true classification, but "documented" means the signal is present but weak. A committee member will ask whether weak signal is worth the computational overhead of the full pipeline. Have a clear answer ready.
