# NATS-Bench-SSS Pipeline Scripts

Scripts for running the zero-cost proxy pipeline on NATS-Bench-SSS (Size Search Space).

## Status
Placeholder — data not yet downloaded.

## What goes here (mirror of scripts/nasbench201/)

| Script | Purpose |
|--------|---------|
| `proxy_utils.py` | Model builder for SSS architectures (channel-width spec, not topology string) |
| `compute_proxy_naswot.py` | NASWOT scores for all SSS architectures |
| `compute_proxy_zenscore.py` | Zen-Score for all SSS architectures |
| `compute_proxy_synflow.py` | SynFlow for all SSS architectures |
| `compute_proxy_param_count.py` | Parameter count for all SSS architectures |
| `transform_proxies.py` | Log transforms (same transforms as NAS-Bench-201) |
| `analyze_distributions.py` | Distribution statistics and plots |
| `validate_ranking_correlations.py` | Spearman rho, Kendall tau, top-K precision |
| `bias_disentanglement.py` | OLS residualisation + partial correlation |

## Key difference from NAS-Bench-201

NAS-Bench-201 architecture strings: `|op~0|+|op~0|op~1|+|op~0|op~1|op~2|`  
NATS-Bench-SSS architecture specs: channel width per layer, e.g. `[8, 16, 32, 32, 64]`

The `proxy_utils.py` model builder needs to be rewritten for SSS. Everything else
(proxy algorithms, transforms, correlation validation) is reusable with updated paths.

## Data location
Place downloaded NATS-Bench-SSS data in: `data/nats_bench_sss/`

## Results location
All outputs go to: `results/nats_bench_sss/{step}/`


Let me read the key scripts to understand exactly what needs to be adapted before writing the plan. 

Read [](file:///f%3A/Thesis/Experimentation/scripts/nasbench201/compute_proxy_naswot.py#1-1), lines 1 to 80

Read [](file:///f%3A/Thesis/Experimentation/scripts/nasbench201/compute_proxy_zenscore.py#1-1), lines 1 to 80

Read [](file:///f%3A/Thesis/Experimentation/scripts/nasbench201/transform_proxies.py#1-1), lines 1 to 60

Read [](file:///f%3A/Thesis/Experimentation/scripts/nasbench201/proxy_utils.py#80-80), lines 80 to 120

Read [](file:///f%3A/Thesis/Experimentation/scripts/nasbench201/proxy_utils.py#150-150), lines 150 to 208

Now I have a complete picture. Here is the full plan:

---

## NATS-Bench SSS Experimental Pipeline Plan

### Guiding constraints from NAS-Bench-201 experience

Before any step: the NAS-Bench-201 lessons that directly constrain SSS decisions are:
- OLS residual ρ was optimistic for all three proxies due to discrete param_count. SSS will be worse — param_count is a nonlinear function of 5 × 8-valued positions, producing a highly multi-modal discrete distribution. Partial rank Spearman is the only credible primary method.
- SynFlow's ρ of 0.164 was entirely absorbed by param_count. In SSS, which is a pure size search space, the capacity confound is even stronger. SynFlow should be expected to be excluded. Compute it anyway for the ablation, but budget no architectural decisions on it.
- NASWOT and ZenScore may not survive debiasing on SSS. SSS is definitionally a capacity search space — the only thing varying is channel widths. Activation diversity proxies are expected to be even more heavily confounded with capacity here than on NAS-Bench-201 where partial ρ was only 0.15–0.19. Plan explicitly for the exclusion outcome before running Step 5.
- Negation direction for NASWOT and ZenScore was empirically confirmed on NAS-Bench-201 due to BatchNorm behavior on skip-heavy architectures. SSS has no skip/none operations — every edge is a convolution. The sign of the raw correlation may differ. Do not hardcode negation — measure first.

---

### Step 0 — Structural Audit (`inspect_sss_structure.py`)

**Purpose**: Answer every question the downstream pipeline depends on, before writing any proxy extraction code. All 6 questions below must be answered and documented in the progress note before proceeding.

**Question 1 — GT accuracy key verification**
Load 5 architectures (indices 0, 1000, 8192, 16383, 32767) from the simple archive. Navigate to `arch2infos[idx]['90']['all_results'][('cifar10', 777)]`. Print the accuracy value. Confirm: (a) values fall in 60–95%, (b) the key actually exists for all 5, (c) the value is test accuracy not validation (the key `'cifar10'` vs `'cifar10-valid'` distinction matters — `'cifar10'` should be the held-out test set).

*What can go wrong*: The result object under `('cifar10', 777)` may be a custom class (not a plain dict) with a method to retrieve accuracy rather than a direct field. The inspect script showed `all_results` values were `<truncated>`. Write a loop that prints `type(result_obj)` and `dir(result_obj)` for the first architecture to know how to extract the scalar accuracy value.

**Question 2 — Accuracy distribution shape**
Load GT accuracy for all 32,768 architectures. Plot a histogram with 100 bins. Record: min, max, mean, std, and whether there is a visible gap separating failed architectures (near 10% as in NAS-Bench-201) from competitive ones. On NAS-Bench-201 there were ~300 degenerate architectures near 10% — on SSS, since all operations are convolutions (no `none` operations that cause failure), this cluster may not exist. If it does not exist, the global ρ values in Step 4 will be lower than NAS-Bench-201's because there are no easy-to-rank dead architectures inflating the correlation. Document this explicitly — it affects how you interpret the proxy validation results.

**Question 3 — Architecture space structure verification**
Parse 100 random arch strings. Confirm: (a) each string has exactly 5 colon-separated integers, (b) all values are in `{8, 16, 24, 32, 40, 48, 56, 64}` (8 choices × 5 positions = 32,768 total ✓), (c) no position has zero variance. Also confirm that the model accessed via the pickle file has no operation-type variation — every layer is a Conv2d. This last point is what makes `proxy_utils_sss.py` structurally different from proxy_utils.py: the NAS-Bench-201 builder had to parse operation strings (`nor_conv_3x3`, `skip_connect`, etc.); the SSS builder only needs to parse 5 integers.

**Question 4 — Channel-width position vs param_count as covariate**
Parse all 32,768 arch strings into a (32768, 5) matrix of channel widths. Compute Spearman ρ between each of the 5 positions and GT accuracy, and between total param_count and GT accuracy. The key question: does any single position predict GT accuracy better than total param_count? If yes, that position carries information not captured by total parameter count and the OLS regression must include it as an additional covariate alongside param_count. If total param_count dominates all 5 positions, single-covariate OLS transfers from NAS-Bench-201 without modification.

*What can go wrong*: The 5 channel-width positions are not independent — later stages depend on earlier ones in a ResNet-style skip connection pattern. Position 5 (the final channel count) will likely correlate most strongly with accuracy because it determines the feature dimensionality going into the classifier. This may make param_count a sufficient summary statistic even if individual positions have high individual ρ.

**Question 5 — FLOPs availability**
Check whether the benchmark stores FLOPs. Navigate into a result object and look for a cost or flops field. The NATS-Bench paper documents that cost info (flops, params, latency) is queryable via the API — check whether it is embedded in the pickle data or only available via the API's separate computation method. If FLOPs is available as a precomputed value, compute its ρ with param_count and with GT accuracy to determine whether it adds independent bias signal.

**Question 6 — Raw NASWOT/ZenScore correlation direction**
This requires running the proxy on a small subset (100–500 architectures) before committing to the full extraction run. Build SSS models for 200 random architectures, compute NASWOT and ZenScore scores, load their GT accuracies, and compute Spearman ρ. Record the sign. If positive → no negation needed. If negative → negation applies as in NAS-Bench-201. Record this explicitly in the progress note as the direction decision before running the full extraction.

---

### Step 1 — Proxy Computation

**New file required: `proxy_utils_sss.py`**

The NAS-Bench-201 proxy_utils.py has a `NASBench201Cell` class that parses operation strings and builds a cell-op DAG. This is completely irrelevant to SSS. `proxy_utils_sss.py` needs a new model builder that takes an arch string like `'16:32:64:32:64'` and builds a sequential ResNet-style network with those channel widths per stage.

The SSS network structure (from NATS-Bench paper): a 5-stage network where each stage has a fixed operation type but varying channel width. The key reference is the channel widths the paper uses — the model must be structurally consistent with how the benchmark architectures were trained so that NASWOT/ZenScore scores are computed on the actual architecture shape, not an approximation. The audit step (Question 3) should confirm this by checking what the actual architecture looks like from the pickle data.

*Risk*: If the SSS pickle data does not include the model definition (it stores results, not weights or topology beyond the arch string), the model builder must be reconstructed from the NATS-Bench paper's architecture description. This is the same situation as NAS-Bench-201 — the model was rebuilt from the specification, not loaded. The difference is that SSS's topology is simpler (no operation search, only width search) so the reconstruction is more straightforward.

**Four scripts to write** (one per proxy, mirroring the NAS-Bench-201 set):

`compute_proxy_param_count_sss.py` — simplest. Parse the 5 integers from the arch string, build the model, count parameters. Runtime: minutes for all 32,768 architectures.

`compute_proxy_naswot_sss.py` — use the identical multi-layer Conv2d hooking implementation from compute_proxy_naswot.py. The only change is the data loading (from `{idx}.pickle` files instead of `.pth` chunk files) and the model builder (from `proxy_utils_sss.py` instead of proxy_utils.py). The hook logic, covariance trace computation, and aggregation are proxy-algorithm logic — they transfer unchanged.

`compute_proxy_zenscore_sss.py` — same transfer pattern as NASWOT. 4-sample multi-layer hooking with variance computation transfers directly. Data loading and model builder change.

`compute_proxy_synflow_sss.py` — transfer the Tanaka et al. algorithm unchanged. Document that this is for ablation/documentation only — expected to be excluded in Step 5. Runtime may be longer than param_count since SynFlow requires a gradient computation.

**Data loading change**: NAS-Bench-201 scripts load from `data/nasbench201/chunks_clean/arch2infos/{idx}.pth` via `torch.load()`. SSS scripts load from `data/nats_bench_sss/NATS-sss-v1_0-50262-simple/{idx}.pickle` via `pickle.load()`. Each entry is an `OrderedDict` with keys `'01'`, `'12'`, `'90'`. The arch string is at `entry['90']['arch_str']` (confirmed by the inspect output).

**Runtime estimate**: 32,768 architectures is ~2× NAS-Bench-201 (15,625). NASWOT and ZenScore on NAS-Bench-201 ran on CPU. Expect proportionally longer. Run param_count first (fastest) to validate the data loading pipeline before committing to the full NASWOT/ZenScore runs.

---

### Step 2 — Log Transformation

`transform_proxies_sss.py` is a near-copy of transform_proxies.py with three targeted changes:
1. Input directory: raw_proxy_scores
2. Output directory: transformed_proxy
3. Negation decision for NASWOT and ZenScore: determined by Step 0 Question 6, not hardcoded. If the audit shows positive raw correlation → `log(score + ε)`. If negative → `-log(score + ε)`.

The transform_proxies.py script has the negation direction hardcoded in the docstring as a rationale specific to NAS-Bench-201's skip/none architecture behavior. That rationale does not transfer to SSS. The SSS version's docstring must state the direction decision explicitly with the empirical evidence from the audit as justification.

---

### Step 3 — Distribution Analysis

`analyze_distributions_sss.py` — copy with path changes. No logic changes. The plots will look different because:
- No `none`-dominated degenerate cluster expected → distributions should be smoother, more unimodal
- Channel-width search space → param_count distribution will be more spread (8 discrete values per position, product of 5 positions → many distinct capacity levels), potentially less multi-modal than NAS-Bench-201's 3–4 peaks

Document explicitly in the progress note whether the distributions are better-behaved (more Gaussian-like) than NAS-Bench-201. This matters for interpreting why OLS vs partial rank Spearman agree or disagree in Step 5.

---

### Step 4 — Ranking Correlation Validation

`validate_ranking_correlations_sss.py` — copy with path changes. GT accuracy loading changes: navigate to `entry['90']['all_results'][('cifar10', 777)]` and extract the accuracy scalar (using whatever access pattern the audit reveals for the result object type).

Key interpretation note: If there is no degenerate cluster near 10% on SSS, the global ρ values will be lower than NAS-Bench-201's (where ~300 architectures near 10% inflated all correlations). A global ρ of 0.4 on SSS may be equivalent in information content to ρ = 0.55 on NAS-Bench-201. Document this explicitly in the progress note when recording the Step 4 results — do not interpret lower ρ as worse proxies without accounting for benchmark difficulty.

---

### Step 5 — Bias Disentanglement

This is the highest-risk step and the one that diverges most from NAS-Bench-201.

**Covariate selection** (informed by Step 0 Question 4):
- If param_count dominates all 5 individual positions → single-covariate OLS as in NAS-Bench-201
- If any channel-width position adds independent predictive power → multi-covariate OLS: `P ~ b0 + b1*param_count + b2*ch_position_k`

**Primary method**: partial rank Spearman from the start. The OLS residual method is computed and reported for methodological completeness, but is not the primary result and is not used for the decision. This is the lesson from NAS-Bench-201 stated explicitly in the code and documentation.

**Three-band decision rule** applied as before. But before running, document the hypothesis: given that SSS is a pure capacity search space, the expected outcome is that NASWOT and ZenScore have higher param_count R² than on NAS-Bench-201 (which was already 43–46%), and lower partial ρ (which was 0.15–0.19). The most likely outcome is exclusion of all three proxies.

**Contingency plan — if all three proxies are excluded**:
Do not proceed to PCA. The pipeline collapses to param_count alone as a predictor. This is a legitimate research finding: SSS is so dominated by capacity variation that no zero-cost activation proxy retains independent signal after debiasing. Document it in the progress note as a benchmark characterization finding. The thesis narrative becomes a comparison between NAS-Bench-201 (where partial signal survives) and SSS (where it does not), which motivates the benchmark selection criteria for future proxy evaluation work. Then have the supervisor conversation about whether to add a third benchmark or new proxies before proceeding to MLP training.

**Contingency plan — if one proxy survives but not the other**:
PCA is still possible but degenerates to a 2-feature input (param_count + one residual). PCA of a 2×2 matrix produces 2 PCs. Document the reduced feature set and proceed.

---

### Step 6 — PCA Whitening (conditional)

Only reached if at least one proxy survives Step 5.

`pca_whitening_sss.py` — copy with path changes and a parameter change for the input feature list (which depends on Step 5 outcomes). The PCA logic, validation checks, and output structure transfer unchanged.

The eigenvalue structure will differ from NAS-Bench-201 (1.559/1.000/0.441). On SSS, if NASWOT and ZenScore are more heavily confounded with param_count, their residuals will have lower variance, and the contrast PC (PC3 on NAS-Bench-201) may have near-zero eigenvalue. Document whatever eigenvalue structure emerges without comparison to NAS-Bench-201 values as a target.

---

### Step 7 — Surrogate MLP Training (conditional)

Only reached if Step 6 completes. Then train the MLP on SSS using the same architecture (3→64→32→1), loss function (pairwise ranking), and evaluation metrics (Spearman ρ, Kendall τ, top-K precision) as planned for NAS-Bench-201 Step 7. At this point run NAS-Bench-201 Step 7 in parallel and produce a joint comparison table.

---

### Risk register

| Risk | Probability | Response |
|---|---|---|
| All activation proxies excluded at Step 5 | High | Stop, document as finding, consult supervisor |
| NASWOT/ZenScore negation direction positive (no negation) | Medium | Apply `log(x + ε)` without negation — measure determines this |
| Result object in pickle is a custom class with non-obvious accuracy accessor | Medium | The Step 0 audit resolves this before any pipeline code is written |
| Multi-covariate OLS needed (individual channel widths add signal) | Low-Medium | Step 0 Question 4 determines this; handled in Step 5 covariate design |
| No degenerate cluster → global ρ lower than expected → misleading comparison to NAS-Bench-201 | Certain | Document explicitly in Steps 4 and 5; normalize interpretation |
| param_count distribution more multi-modal than NAS-Bench-201 → OLS even less reliable | Likely | Already handled by using partial rank Spearman as primary throughout |