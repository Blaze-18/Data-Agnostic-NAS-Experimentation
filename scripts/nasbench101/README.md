# NAS-Bench-101 Pipeline Plan

**Benchmark**: NAS-Bench-101 (Ying et al. 2019)
**Dataset file**: `data/nasbench101/nasbench_full.tfrecord` (2.1 GB)
**Architectures**: 423,624 unique (7-node DAGs, 5 operations, max 9 edges)
**GT**: Average final_test_accuracy over 3 seeds at 108 epochs on CIFAR-10
**Feasibility verdict**: R^2(log_param -> GT) = 0.047 -- VIABLE

---

## Feasibility Summary (pre-computation results)

| Metric | Value | Interpretation |
|---|---|---|
| R^2 (size -> GT) | **0.047** | Size explains only 4.7% of variance -- proxies have full room |
| Spearman rho (param, GT) | 0.435 | Moderate size correlation -- not dominant |
| GT std | 5.80% | Large variation -- easy to rank across the full space |
| GT range | 9.98% -- 94.32% | Full range including degenerate archs |
| GT skewness | -8.07 | Heavy left tail (degenerate archs cluster near 10%) |
| Top-1% std | 0.12% | Tight at the top -- top-K evaluation at 5%+ is more meaningful |

**Key difference from NAS-Bench-201**: R^2=0.047 vs 0.157.
Size is almost irrelevant on NAS-Bench-101. The methodology has maximum room to work.

**Key difference from NATS-Bench SSS**: R^2=0.047 vs 0.789.
No structural dominance problem. Proxies must earn their signal.

---

## Critical facts about NAS-Bench-101 architecture format

Every architecture is a DAG defined by:
- `module_adjacency`: upper-triangular binary matrix (7x7)
- `module_operations`: list of 7 strings, first always INPUT, last always OUTPUT
- Available intermediate ops: `conv3x3-bn-relu`, `conv1x1-bn-relu`, `maxpool3x3`
- Max 9 edges enforced by benchmark construction

**Building a PyTorch model from a NAS-Bench-101 spec is the hardest part of this pipeline.**
The NAS-Bench-201 `proxy_utils.py` built models from an arch string -- NAS-Bench-101
requires building a DAG network from an adjacency matrix + ops list.
A `NASBench101Cell` builder must be written as `proxy_utils_101.py` before any proxy scripts.

**Isomorphic deduplication is already handled by the nasbench API.**
`nb.hash_iterator()` returns canonical hashes only -- no duplicate architectures.

---

## API loading pattern (required boilerplate for every script)

```python
import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"  # REQUIRED
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from nasbench import api

nb = api.NASBench("data/nasbench101/nasbench_full.tfrecord")
# Loading takes ~390 seconds -- do not load more than once per script run
```

Three patches were applied to make this work with Python 3.12 + TF 2.21:
1. `api.py`: commented out `from nasbench.lib import evaluate` (TF1 class removed in TF2)
2. `api.py`: changed `tf.python_io.tf_record_iterator` to `tf.compat.v1.io.tf_record_iterator`
3. `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` env var (protobuf version mismatch)

**GT access pattern** (verified):
```python
fixed, computed = nb.get_metrics_from_hash(h)
params = fixed["trainable_parameters"]
adjacency = fixed["module_adjacency"]   # numpy int8 array (7,7)
operations = fixed["module_operations"] # list of 7 strings
runs = computed[108]                    # list of 3 dicts (one per seed)
test_acc = float(np.mean([r["final_test_accuracy"] for r in runs]))
# Note: stored as fraction (0.0-1.0), multiply by 100 for percentage
```

---

## Lessons from NAS-Bench-201 and NATS-Bench SSS (applied here)

| Lesson | How it is applied |
|---|---|
| Load API once, dump all data to .npy upfront | Step 0 saves GT + params to .npy so later steps never reload the TFRecord |
| Proxy compute scripts must checkpoint progress | 423k architectures -- if a compute script crashes at 400k, restart from checkpoint |
| NASWOT/ZenScore: hook ALL Conv2d layers, not just first | Multi-layer hooking achieved 100% coverage vs 26% single-layer |
| SynFlow: all-ones input + weight linearization | Random input degrades rho to ~0.17 on NAS-Bench-201; same applies here |
| Partial rank Spearman is the credible method | OLS residual rho was inflated on NAS-Bench-201 due to discrete param_count -- use partial rank as primary |
| Do not use MSE loss for surrogate MLP | Pairwise ranking loss preserves rank order; MSE optimizes point accuracy |
| Top-1% precision is unreliable at tight accuracy windows | Top-1% std = 0.12% -- report top-5% and top-10% precision as primary metrics |
| ASCII-only in all script files | No Unicode > 127 -- Windows cp1252 encoding crash risk |
| NAS-Bench-101 specific: degenerate architectures exist | ~8% of architectures cluster near 10% accuracy (no-op/skip dominated) -- these inflate global rho; report rho on competitive subset (>50% accuracy) separately |

---

## NAS-Bench-101 specific risks

**Risk 1: Skip connection pathology**
High-performing NAS-Bench-101 architectures often have many skip connections.
NASWOT and ZenScore measure activation diversity -- skip connections reduce diversity.
After debiasing, partial rho for NASWOT/ZenScore may be moderate or negative in
regions with many skips. This is documented in the NASWOT paper. Do not over-interpret
if NASWOT partial rho is lower than on NAS-Bench-201.

**Risk 2: Nonlinear size relationship**
OLS assumes linear log(param_count) -> GT. NAS-Bench-101 has more diverse depth and
width variation than NAS-Bench-201. Check residual plots after Step 5 for curvature.
If clearly nonlinear, note it in the results -- partial rank Spearman handles this
without modification.

**Risk 3: Compute time**
- SynFlow: ~30 min for 32k (SSS) -> estimated ~4-6 hours for 423k
- NASWOT: similar to SynFlow
- ZenScore: similar to SynFlow
- Strategy: run all three compute scripts sequentially overnight; each must checkpoint

**Risk 4: Model builder complexity**
The DAG-based model is harder to build than the NAS-Bench-201 cell string model.
Verify the model builder produces correct parameter counts before running any proxy.
Cross-check: `model_param_count == fixed["trainable_parameters"]` for 10 random archs.

---

## Pipeline Steps

### Step 0 -- Data Audit and GT Extraction
**Script**: `step0_audit_101.py`
**Output**: `results/nasbench101/audit/`

Goals:
- Load TFRecord, iterate all 423,624 architectures
- Save GT array: `gt_accuracies.npy` shape (423624,) float32, values in [0,100]
- Save param counts: `param_counts.npy` shape (423624,) float64
- Save arch hashes: `arch_hashes.npy` shape (423624,) -- ordered list of hash strings
- Save adjacency + ops for all archs: `arch_specs.pkl` -- dict hash -> {adj, ops}
- Compute and save audit_summary.json with:
  - GT distribution stats (mean, std, min, max, skewness, percentiles)
  - Param count distribution stats
  - Number of unique ops distributions
  - Degenerate arch count (accuracy < 20%)
  - R^2 reconfirmation (should match 0.047)

**Why**: All downstream steps load .npy files, never the TFRecord again.
The 390-second load happens exactly once.

---

### Step 1 -- Proxy Computation
**Scripts**: `compute_proxy_param_count_101.py`, `compute_proxy_synflow_101.py`,
            `compute_proxy_naswot_101.py`, `compute_proxy_zenscore_101.py`
**Shared utility**: `proxy_utils_101.py` -- NASBench101Net model builder
**Output**: `results/nasbench101/raw_proxy_scores/`
            `param_count.npy`, `synflow.npy`, `naswot.npy`, `zenscore.npy`
            Each shape (423624,) with values in original score space

**Post-compute fix (applied May 9, 2026)**:
`fix_naswot_normalisation.py` corrects a normalisation bug in `compute_proxy_naswot_101.py`
where NASWOT traces were divided by the number of Conv2d layers *in that architecture*
(variable, 1–48) instead of the global maximum (`MAX_CONV = 48`). This inflated scores
for shallow architectures. The fix divides all traces by the fixed constant 48 and
re-saves `naswot.npy`. The original buggy scores are backed up as `naswot_original_mean.npy`.
If re-running the pipeline from scratch, the bug is already fixed in `compute_proxy_naswot_101.py`
(divide by `MAX_CONV = 48` is now hardcoded) — `fix_naswot_normalisation.py` only needs
to be run if restoring from an existing checkpoint that used the old mean normalisation.

**proxy_utils_101.py must implement**:
```
NASBench101Net(adjacency, operations) -> nn.Module
```
The network must:
- Build a DAG where each intermediate node aggregates outputs from all predecessor nodes
- Apply the correct operation at each node (conv3x3/conv1x1/maxpool3x3)
- Use a stem conv (3->128, 3x3) before the cell and a global avgpool + FC after
- Support a stack of cells (typically 3 stacks of 3 cells each = 9 cells)
- Input size: (1, 3, 32, 32) for SynFlow; (8, 3, 32, 32) for NASWOT/ZenScore

**param_count script**: reads from `results/nasbench101/audit/param_counts.npy` directly
-- no model needed, already extracted in Step 0.

**Checkpoint pattern for all three compute scripts**:
```python
CHECKPOINT_FILE = "results/nasbench101/raw_proxy_scores/{proxy}_checkpoint.json"
# Save progress every 1000 architectures
# On restart, skip already-computed hashes
# Final: delete checkpoint file after saving .npy
```

**Estimated compute times (CPU, based on SSS rates)**:
- SynFlow: 4-6 hours
- NASWOT: 4-6 hours
- ZenScore (4 random draws): 5-8 hours

Run all three overnight in separate terminals.

---

### Step 2 -- Log Transformation
**Script**: `transform_proxies_101.py`
**Input**: `results/nasbench101/raw_proxy_scores/`
**Output**: `results/nasbench101/transformed_proxy/`

Transformations (same logic as NAS-Bench-201):
- param_count: log(x)
- synflow: log(x + 1e-8), check for zeros/negatives first
- naswot: negate then log(-x + epsilon) if raw correlation is negative;
         OR log(x) if positive -- **check raw Spearman sign first**
- zenscore: same sign check as naswot

**Do not assume negation direction from NAS-Bench-201 results.**
NAS-Bench-101 is a different architecture space. Run Step 4 correlation check on
raw values before deciding negation. If raw Spearman rho is negative, negate before log.

---

### Step 3 -- Distribution Analysis
**Script**: `analyze_distributions_101.py`
**Input**: `results/nasbench101/transformed_proxy/`
**Output**: `results/nasbench101/proxy_distribution/`

Compute per proxy: mean, std, skewness, kurtosis, min/max, histogram plots.
Flag any proxy where log-transform does not normalize (skewness > 2 or < -2).
Save `stats_summary.json` with all values for documentation.

---

### Step 4 -- Ranking Correlation Validation
**Script**: `validate_ranking_correlations_101.py`
**Input**: transformed proxy scores + GT from audit
**Output**: `results/nasbench101/proxy_validation/`
            `correlation_results.json`, scatter plots, pairwise heatmap

Compute: Spearman rho, Kendall tau, top-5% and top-10% precision (NOT top-1%).
**Report two rho values per proxy**:
1. Global rho (all 423,624 archs) -- will be inflated by degenerate archs
2. Competitive rho (archs with GT > 50% only, ~380k archs) -- the honest number

This distinction is essential for the thesis. NAS-Bench-101 has ~8% degenerate
architectures. Every proxy trivially identifies them, inflating global rho by design.

---

### Step 5 -- Bias Disentanglement
**Script**: `bias_disentanglement_101.py`
**Input**: transformed proxies, GT from audit
**Output**: `results/nasbench101/debiased_proxy/`

Identical method to NAS-Bench-201:
- OLS: regress each proxy on log(param_count), save residuals
- Compute OLS-residual Spearman AND partial-rank Spearman
- If |delta| > 0.05: use partial-rank as primary (same threshold)
- Decision: rho >= 0.30 KEEP; 0.10-0.30 KEEP_DOCUMENTED; < 0.10 EXCLUDE
- Save partial_correlations.json with decisions

**Expected outcome** (informed estimate, not a given):
- SynFlow: unknown -- on NAS-Bench-201 it collapsed to -0.002; may survive here
  since gradient flow is sensitive to DAG depth/connectivity, not just size
- NASWOT: may be moderate (skip connection pathology applies)
- ZenScore: similar to NASWOT

The correct attitude: measure and report whatever the data says.
Do not try to match NAS-Bench-201 results.

---

### Step 6 -- PCA Whitening
**Script**: `pca_whitening_101.py`
**Input**: debiased residuals from Step 5 + param_count_transformed
**Output**: `results/nasbench101/pca_whitening/`
            `whitened_features.npy` shape (423624, k) where k = number of kept proxies + 1

Identical method to NAS-Bench-201:
- Input matrix: [param_count_transformed, {kept_proxy}_residuals...]
- StandardScaler -> PCA with whiten=True -> retain PCs until 99% variance
- Validate: |col_mean| < 1e-5, max|Cov(Z) - I| < 1e-4

**Note on input size**: if only 1 proxy survives Step 5 (e.g., only SynFlow),
the input is (423624, 2) -- param_count + 1 residual. PCA still applies; the
whitening step guarantees unit variance on both dimensions.

---

### Step 7 -- Surrogate MLP
**Script**: `train_mlp_101.py`
**Input**: `results/nasbench101/pca_whitening/whitened_features.npy`
**Output**: `results/nasbench101/surrogate_mlp/`

Architecture: k -> 64 -> 32 -> 1 (k = number of whitened features, typically 2-4)
Loss: pairwise ranking loss (not MSE)
Split: 80% train / 10% val / 10% test (fixed seed=42)
Optimizer: Adam, lr=1e-3, early stopping on val ranking loss (patience=20)

**Ablation table (required for thesis)**:
Run four variants and record all metrics for each:
1. param_count only (size-only baseline)
2. Best raw proxy only (no debiasing, no PCA)
3. PCA on raw proxies (PCA without debiasing)
4. Full pipeline (whitened debiased features) -- the proposed method

Evaluation metrics:
- Spearman rho (global and competitive subset)
- Kendall tau
- Top-5% precision
- Top-10% precision
- (Report top-1% for completeness but do not use as primary metric)

---

## File naming conventions

All scripts follow the pattern: `{step_name}_101.py`
All output directories mirror `results/nasbench201/` exactly.
All .npy files are float32 unless noted. arch_hashes.npy is object dtype.

---

## Execution order

```
Step 0:  python scripts/nasbench101/step0_audit_101.py           (~7 min)
Step 1a: python scripts/nasbench101/compute_proxy_synflow_101.py  (~5 hr)
Step 1b: python scripts/nasbench101/compute_proxy_naswot_101.py   (~5 hr)  [parallel or overnight]
Step 1c: python scripts/nasbench101/compute_proxy_zenscore_101.py (~7 hr)  [parallel or overnight]
Step 2:  python scripts/nasbench101/transform_proxies_101.py      (~1 min)
Step 3:  python scripts/nasbench101/analyze_distributions_101.py  (~1 min)
Step 4:  python scripts/nasbench101/validate_ranking_correlations_101.py (~2 min)
Step 5:  python scripts/nasbench101/bias_disentanglement_101.py   (~5 min)
Step 6:  python scripts/nasbench101/pca_whitening_101.py          (~2 min)
Step 7:  python scripts/nasbench101/train_mlp_101.py              (~10-30 min)
```

Steps 1a, 1b, 1c can be run in parallel in separate terminals if RAM allows
(each loads its own model instances, no shared state). Total RAM for 3 parallel
proxy scripts: ~4-6 GB.

Total wall-clock time: 1 overnight run for Step 1, then Steps 2-7 in under 1 hour.

---

## Decision gate: what to do if Step 5 partial rhos are all low

If all proxies return partial rho < 0.10 after debiasing (unlikely given R^2=0.047,
but possible due to skip connection pathology), the conclusion is:
- NAS-Bench-101 proxies do not carry size-independent signal
- This would be a genuinely novel finding: even on a benchmark where size is irrelevant,
  activation-diversity proxies cannot rank architectures
- Report this as a finding, not a failure; compare to NAS-Bench-201 and SSS
- Step 7 MLP would still run on raw features as an ablation to demonstrate the point

The minimum viable thesis contribution is the cross-benchmark comparison table:
SSS (size-dominated) vs NAS-Bench-201 (operation-diverse, fixed topology) vs
NAS-Bench-101 (DAG-diverse, operation-diverse) -- showing how proxy utility
depends on search space structure. This finding holds regardless of Step 7 outcomes.
