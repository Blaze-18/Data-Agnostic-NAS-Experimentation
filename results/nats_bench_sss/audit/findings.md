## Audit Analysis

All 6 questions passed. Here is what the results mean for every downstream step.

---

### Finding 1 — The accuracy distribution fundamentally changes the interpretation baseline (Q2)

The entire SSS search space sits between **79.79% and 93.65%** (std = 1.27%). There are zero failed architectures. This is the single most important structural difference from NAS-Bench-201:

- NAS-Bench-201 had ~300 architectures near 10% — those dead architectures were trivially easy to rank to the bottom, mechanically inflating every proxy's ρ
- SSS has no such cluster — every architecture is functional, and the proxy must discriminate within a 13.86% range with std = 1.27%
- The top 1% spans only 93.05–93.65% (0.6% absolute spread)

**Consequence**: any global ρ on SSS is a harder number to achieve than the same ρ on NAS-Bench-201. A proxy achieving ρ = 0.45 on SSS is doing more discriminative work than one achieving ρ = 0.55 on NAS-Bench-201. This must be stated explicitly when comparing the two benchmarks.

---

### Finding 2 — Capacity dominance is far stronger than on NAS-Bench-201 (Q4)

| Covariate | SSS ρ | NAS-Bench-201 equivalent |
|---|---|---|
| Σch (sum of widths) | **0.925** | — |
| Σch² (param proxy) | **0.857** | 0.749 |
| Best single position (Pos 2) | 0.527 | — |

Σch explains roughly 85% of accuracy rank variance. On NAS-Bench-201 the OLS R² on param_count was only 15.7% (linear fit) — on SSS, the structural capacity signal is overwhelmingly dominant. The risk that NASWOT and ZenScore have no residual signal after debiasing is high.

One critical note: Σch (ρ=0.925) is a stronger covariate than Σch² (ρ=0.857). For the OLS in Step 5, the bias disentanglement script must use **actual computed param_count** (from running the model), not either proxy. Actual param_count is ∝ Σ(c_i × c_{i+1}) for adjacent conv stages — neither purely Σch nor Σch². The audit Σch result is a flag that capacity dominance is extreme, not a recommendation to use Σch as the OLS predictor.

---

### Finding 3 — The 5 positions are perfectly orthogonal (Q3/Q4)

The pairwise position correlation matrix is exactly the identity. This is the 8⁵ full factorial design by construction — each position is varied independently. This confirms single-covariate OLS is correct: the positions don't collude, and their collective effect is already captured by total param_count.

---

### Finding 4 — No negation for NASWOT/ZenScore (Q6)

Raw NASWOT ρ = **+0.463**. Larger covariance trace = wider channel networks = better accuracy. This is the opposite of NAS-Bench-201, where skip/none operations caused high-variance activations that correlated negatively with accuracy.

Transform for Step 2: `log(score + ε)` — no negation. This decision is locked in by the audit.

---

### Finding 5 — FLOPs not in the pickle data (Q5)

FLOPs are not stored at any epoch-data level. Pursuing FLOPs as a secondary covariate would require running the NATS-Bench API's cost computation on all 32,768 architectures. Given that Σch already explains 85% of variance and the single-covariate recommendation is solid, this is not worth the complexity.

---

### Pipeline decisions locked in by the audit

| Decision | Value | Source |
|---|---|---|
| GT accuracy key | `('cifar10', 777)` in `'90'` slot, dict access | Q1 |
| Degenerate cluster | None — lower global ρ expected vs NAS-Bench-201 | Q2 |
| OLS covariate | Actual param_count (single covariate) | Q4 |
| Primary debias method | Partial rank Spearman (not OLS residuals) | Q4 + NAS-Bench-201 lesson |
| NASWOT/ZenScore transform | `log(score + ε)`, NO negation | Q6 |
| FLOPs | Not pursued | Q5 |

