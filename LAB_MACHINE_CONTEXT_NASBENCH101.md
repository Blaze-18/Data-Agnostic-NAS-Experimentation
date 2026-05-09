# NAS-Bench-101 Lab Machine Context
# Experiment: Zero-Cost Proxy Bias Disentanglement on NAS-Bench-101

**Written**: May 8, 2026
**Purpose**: Full context for running the NAS-Bench-101 experiment from scratch on the lab machine.
**Lab machine**: GTX 5070 Ti, 32 GB RAM, Windows (assume same OS)
**Repository root**: clone from git (scripts only -- all data files are gitignored)
**Starting point**: nasbench_full.tfrecord is already on the lab machine. Run everything from Step 0.

---

## 1. What This Experiment Is

This is the third benchmark in a thesis that evaluates whether zero-cost proxies
can rank neural architectures fairly AFTER removing structural size bias.

The core methodology, already validated on two other benchmarks:
  1. Compute 4 zero-cost proxies on all architectures
  2. Log-transform raw scores
  3. Regress out log(param_count) via OLS; keep residuals (bias disentanglement)
  4. PCA whiten the kept features
  5. Train shallow MLP with pairwise ranking loss on whitened features
  6. Compare against baselines in an ablation table

NAS-Bench-101 is the most important benchmark for the thesis because:
  - R^2(param_count -> GT) = 0.047 (size is almost irrelevant)
  - Proxies must earn their signal independently
  - This is the strongest test of the methodology

---

## 2. Cross-Benchmark Context (already completed)

### NAS-Bench-201 (DONE)
- N = 15,625, fixed topology (6 ops, 6 edges)
- R^2 = 0.157 (size explains 15.7% of GT variance)
- Proxies kept after debiasing: NASWOT (partial rho=0.154), ZenScore (partial rho=0.190)
- SynFlow EXCLUDED (partial rho=-0.002 -- zero signal)
- MLP input: param_count + NASWOT_residual + ZenScore_residual (3 features, PCA -> 3 PCs)
- GT: CIFAR-10 at epoch 199

### NATS-Bench SSS (DONE)
- N = 32,768, width-only search (5 channel positions, 8 values each)
- R^2 = 0.789 (size dominates -- methodology revealed this is NOT viable for unbiased ranking)
- Proxies kept: SynFlow ONLY (partial rho=+0.625)
- NASWOT EXCLUDED (partial rho=0.059), ZenScore EXCLUDED (partial rho=0.051)
- MLP input: param_count + SynFlow_residual (2 features, PCA -> 2 PCs)
- Key finding: NASWOT/ZenScore collapse completely when topology is fixed -- they
  measure operation diversity, which does not exist in a width-only space

### NAS-Bench-101 (FULL RUN -- start from Step 0 on this machine)
- N = 423,624, DAG search (variable 4-7 node DAGs)
- R^2 = 0.047 (size is nearly irrelevant -- BEST case for methodology)
- Hypothesis: NASWOT and ZenScore may survive debiasing here (diverse ops exist)
  SynFlow may also carry signal (DAG paths have diverse widths and depths)
- GT: CIFAR-10, mean of 3 seeds at 108 epochs, multiplied by 100 for %

---

## 3. Full Pipeline Run Plan (all steps to be run on this machine)

The experiment starts from scratch. Every step below must be run in order.
No pre-computed results are available -- the lab machine generates everything.

| Step | Script | Status | Notes |
|------|--------|--------|-------|
| 0 | step0_audit_101.py | NOT STARTED | ~390s to load TFRecord; run ONCE |
| 1a | compute_proxy_param_count_101.py | NOT STARTED | Fast (~1 min); reads audit output |
| 1b | gpu/compute_proxy_synflow_gpu.py | NOT STARTED | GPU, ~15-30 min |
| 1c | gpu/compute_proxy_naswot_gpu.py | NOT STARTED | GPU, ~15-30 min |
| 1d | gpu/compute_proxy_zenscore_gpu.py | NOT STARTED | GPU, ~20-40 min |
| 2 | transform_proxies_101.py | NOT STARTED | After all proxies done |
| 3 | analyze_distributions_101.py | NOT STARTED | After Step 2 |
| 4 | validate_ranking_correlations_101.py | NOT STARTED | After Step 2 |
| 5 | bias_disentanglement_101.py | NOT STARTED | After Step 2 |
| 6 | pca_whitening_101.py | NOT STARTED | After Step 5 |
| 7 | train_mlp_101.py | NOT STARTED | After Step 6 |

### API patches (CRITICAL -- must apply before Step 0)
The nasbench package ships broken for Python 3.12 + TF 2.x. Apply 3 patches
before running any script. See Section 5 for full patch instructions.

---

## 4. Data Setup on Lab Machine

The tfrecord is already on the lab machine. Clone the git repo for all scripts.
No result files need to be copied -- everything will be generated from scratch.

### Step: clone the repository
```powershell
git clone <repo-url> Experimentation
cd Experimentation
```

### Step: place the TFRecord in the expected location
The tfrecord must be at:
```
data/nasbench101/nasbench_full.tfrecord   (2.1 GB)
```
Create the directory and move/copy the file there:
```powershell
New-Item -ItemType Directory -Force -Path data\nasbench101
# then copy or move nasbench_full.tfrecord into data\nasbench101\
```

### Step: create empty result directories (scripts create these automatically)
All result directories are created by the scripts themselves via mkdir(parents=True).
Nothing else needs to be set up manually.

### What is gitignored (generated locally, never committed)
- All .npy, .pkl, .pth, .h5 files
- data/ directory contents
- envs/ virtual environment
- results/ binary outputs

### What IS in git (comes from clone)
- All scripts under scripts/nasbench101/
- All scripts under scripts/nasbench101/gpu/
- .gitignore, README.md, this context file

---

## 5. Environment Setup on Lab Machine

### Python version
Python 3.12.x  (same as home machine)

### Create the virtual environment
```powershell
python -m venv envs\nasbench_env
envs\nasbench_env\Scripts\Activate.ps1
```

### Install packages (order matters due to TF compatibility)
```powershell
pip install tensorflow==2.21.0
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install protobuf==3.20.3
pip install numpy scipy scikit-learn matplotlib
pip install nasbench
```

NOTE: Use the CUDA build of PyTorch. Replace cu128 with the correct CUDA version
for the lab machine's driver. Check with: nvidia-smi

### Verify CUDA is available
```powershell
envs\nasbench_env\Scripts\python.exe -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Expected: True, then GPU name.

### NasBench API patches (CRITICAL -- must apply before any script runs)
The nasbench package has 3 TF1->TF2 incompatibilities that must be patched:

File to edit:
  envs\nasbench_env\Lib\site-packages\nasbench\api.py

Patch 1: Comment out the evaluate import (TF1 class removed in TF2)
  # from nasbench.lib import evaluate

Patch 2: Replace the TF record iterator
  OLD: tf.python_io.tf_record_iterator
  NEW: tf.compat.v1.io.tf_record_iterator

Patch 3: Environment variable (set BEFORE importing nasbench in every script)
  os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

Verify the patch worked:
```powershell
$env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"
envs\nasbench_env\Scripts\python.exe -c "
import os; os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION']='python'
from nasbench import api
nb = api.NASBench('data/nasbench101/nasbench_full.tfrecord')
print('Loaded OK, n =', len(list(nb.hash_iterator())))
"
```
Expected: loads in ~390 seconds, prints n=423624

---

## 6. Architecture of the Codebase

```
scripts/nasbench101/
    proxy_utils_101.py         -- CRITICAL shared model builder
    step0_audit_101.py         -- Step 0: run this first
    compute_proxy_param_count_101.py   -- Step 1a: run after Step 0
    compute_proxy_synflow_101.py       -- Step 1b CPU version (DO NOT use on lab machine)
    compute_proxy_naswot_101.py        -- Step 1c CPU version (DO NOT use on lab machine)
    compute_proxy_zenscore_101.py      -- Step 1d CPU version (DO NOT use on lab machine)
    transform_proxies_101.py           -- Step 2
    analyze_distributions_101.py       -- Step 3
    validate_ranking_correlations_101.py  -- Step 4
    bias_disentanglement_101.py        -- Step 5
    pca_whitening_101.py               -- Step 6
    train_mlp_101.py                   -- Step 7
    gpu/                               -- USE THESE ON LAB MACHINE
        compute_proxy_synflow_gpu.py   -- Step 1b GPU version
        compute_proxy_naswot_gpu.py    -- Step 1c GPU version
        compute_proxy_zenscore_gpu.py  -- Step 1d GPU version
    pilot_test/                        -- 20k subset scripts (ignore for full run)

results/nasbench101/
    audit/                             -- generated by Step 0
    raw_proxy_scores/                  -- generated by Steps 1a-1d
    transformed_proxy/                 -- output of Step 2
    proxy_distribution/                -- output of Step 3
    proxy_validation/                  -- output of Step 4
    debiased_proxy/                    -- output of Step 5
    pca_whitening/                     -- output of Step 6
    surrogate_mlp/                     -- output of Step 7
```

---

## 7. proxy_utils_101.py -- Model Builder Details

This is the most critical file. Key facts:

NAS-Bench-101 DAGs have VARIABLE number of nodes (4 to 7, NOT always 7).
This was a bug found during home-PC development and is now FIXED.

Shape distribution in the dataset:
  (7,7): ~84% of architectures
  (6,6): ~14%
  (5,5): ~0.7%
  (4,4): ~0.04%

The fix: NASBench101Cell uses N = adjacency.shape[0] throughout.
Node 0 = INPUT, node N-1 = OUTPUT, nodes 1 to N-2 = intermediate ops.

Network structure for proxy computation (C=16, small and fast):
  stem (Conv2d 3->C) -> 3 stacks of 3 cells each -> AvgPool -> FC(10)
  Stacks use channel doubling: C, 2C, 4C

Verify the model builder is working before launching proxy compute:
```powershell
envs\nasbench_env\Scripts\python.exe scripts/nasbench101/pilot_test/verify_shapes.py
```
Expected output:
  adj_shape=(4, 4)  out=(1, 10)  OK
  adj_shape=(5, 5)  out=(1, 10)  OK
  adj_shape=(6, 6)  out=(1, 10)  OK
  adj_shape=(7, 7)  out=(1, 10)  OK

---

## 8. Running the Full Pipeline (Step-by-Step)

### Step 0: Data audit and GT extraction (run ONCE, ~7 min total)
```powershell
envs\nasbench_env\Scripts\python.exe scripts/nasbench101/step0_audit_101.py
```
This loads nasbench_full.tfrecord (~390s), iterates all 423,624 architectures,
and saves to results/nasbench101/audit/:
  gt_accuracies.npy    (423624,) float32  -- GT in [0,100]%
  param_counts.npy     (423624,) int64
  arch_hashes.npy      (423624,) object   -- ordered hash list
  arch_specs.pkl       dict hash->{adjacency, ops}  (~45 MB)
  audit_summary.json   -- distribution stats and R^2 confirmation

Expected output line: "n_total=423624  n_degenerate_lt20=646"
Do NOT re-run this step after it completes successfully.

### Step 1a: Param count proxy (~1 min)
```powershell
envs\nasbench_env\Scripts\python.exe scripts/nasbench101/compute_proxy_param_count_101.py
```
Copies param_counts from the audit directly. Fast.

### Step 1b/1c/1d: Compute proxies on GPU (open 3 separate terminals)

Terminal 1 -- SynFlow (fastest of the three):
```powershell
cd F:\[YOUR PATH]\Experimentation
envs\nasbench_env\Scripts\python.exe scripts/nasbench101/gpu/compute_proxy_synflow_gpu.py
```

Terminal 2 -- NASWOT:
```powershell
cd F:\[YOUR PATH]\Experimentation
envs\nasbench_env\Scripts\python.exe scripts/nasbench101/gpu/compute_proxy_naswot_gpu.py
```

Terminal 3 -- ZenScore (slowest, 4x samples):
```powershell
cd F:\[YOUR PATH]\Experimentation
envs\nasbench_env\Scripts\python.exe scripts/nasbench101/gpu/compute_proxy_zenscore_gpu.py
```

Each script:
- Prints GPU name and VRAM at startup (confirm it found the 5070 Ti)
- Logs progress every 5000 architectures with ETA
- Checkpoints to JSON every 1000 architectures (safe to interrupt and resume)
- Outputs to results/nasbench101/raw_proxy_scores/{proxy}.npy
- Expected time on GTX 5070 Ti: roughly 15-30 min each

Run all three in parallel (they write to different files, no conflicts).

### Steps 2-7: Analysis pipeline (run sequentially after all proxies complete)

```powershell
# All from the repository root
$PYTHON = "envs\nasbench_env\Scripts\python.exe"

& $PYTHON scripts/nasbench101/transform_proxies_101.py
& $PYTHON scripts/nasbench101/analyze_distributions_101.py
& $PYTHON scripts/nasbench101/validate_ranking_correlations_101.py
& $PYTHON scripts/nasbench101/bias_disentanglement_101.py
& $PYTHON scripts/nasbench101/pca_whitening_101.py
& $PYTHON scripts/nasbench101/train_mlp_101.py
```

Each step takes under 5 minutes. Total for Steps 2-7: ~20-30 minutes.

### Checkpoint recovery (if a proxy script is interrupted)
Each GPU compute script auto-resumes from checkpoint:
  results/nasbench101/raw_proxy_scores/{proxy}_checkpoint.json
Just re-run the same command. It will print "Resumed from checkpoint at idx=N".

---

## 9. Expected Proxy Behavior and Key Concerns

### What we expect based on the two completed benchmarks

| Proxy | NAS-Bench-201 partial rho | SSS partial rho | NAS-Bench-101 prediction |
|-------|--------------------------|-----------------|--------------------------|
| SynFlow | -0.002 (EXCLUDED) | +0.625 (KEPT) | Unknown -- DAG paths add signal? |
| NASWOT | +0.154 (KEPT) | +0.059 (EXCLUDED) | Likely positive -- op diversity exists |
| ZenScore | +0.190 (KEPT) | +0.051 (EXCLUDED) | Likely positive -- same reason |

### Key concern 1: Degenerate architecture cluster
~646 architectures (0.15%) have GT accuracy < 20% (near-random, failed to train).
These will have extreme proxy values in some cases.
The scripts report BOTH:
  - Global Spearman rho (all 423,624 architectures)
  - Competitive rho (architectures with GT > 50%)
Focus on the competitive rho for the thesis -- the global rho is inflated
by the easy separation of degenerate architectures from good ones.

### Key concern 2: SynFlow near-zero scores
On NAS-Bench-101, some architectures may have very small SynFlow scores if
they have many maxpool3x3 operations (no learnable weights -> no gradient flow).
transform_proxies_101.py uses log(x + 1e-8) which handles this safely.
Check the valid count in the SynFlow output: "valid=N/423624"
If valid < 95%, investigate why before proceeding to Step 2.

### Key concern 3: NASWOT/ZenScore signal direction
On NAS-Bench-201, NASWOT and ZenScore required NEGATION before log-transform
because high activation diversity correlated with skip-dominated (bad) architectures.
On NAS-Bench-101, the direction is UNKNOWN until Step 2 runs.
transform_proxies_101.py auto-detects the sign via raw Spearman rho against GT.
Check transform_summary.json after Step 2 to see whether negation was applied.

### Key concern 4: OLS linearity assumption in Step 5
On NAS-Bench-201, the OLS residual method gave inflated rho vs partial-rank
Spearman because param_count had a discrete (~28 unique values) distribution.
On NAS-Bench-101, param_count has MUCH more continuous variation (DAG paths
produce a wide continuous distribution of parameter counts).
Both methods (OLS residual rho AND partial-rank rho) are reported.
If they disagree by > 0.05, partial-rank Spearman is the primary result.

### Key concern 5: Number of features entering PCA (Step 6)
Step 5 decides which proxies to KEEP based on partial-rank rho thresholds:
  >= 0.30  -> KEEP
  0.10-0.30 -> KEEP_DOCUMENTED (kept but noted)
  < 0.10   -> EXCLUDE
The number of PCA input features is NOT fixed -- it depends on Step 5 outcomes.
If all 3 proxies pass, PCA gets 4 columns (param_count + 3 residuals).
If only 1 passes, PCA gets 2 columns (param_count + 1 residual).
The MLP in Step 7 reads the PCA output shape dynamically -- it adapts automatically.

---

## 10. Decision Gate: When Are Results Acceptable?

This lab machine run IS the definitive result for the thesis.

After Step 7 completes, check ablation_table.json for the 4 ablation variants:
  size_only:      MLP trained on param_count alone
  best_raw:       MLP trained on the single best raw proxy (highest global rho)
  pca_raw:        MLP trained on PCA of all raw transformed proxies (no debiasing)
  full_pipeline:  MLP trained on whitened debiased features (the proposed method)

Results are acceptable for the thesis if:
  1. At least one proxy survives debiasing with partial-rank rho > 0.10
  2. full_pipeline Spearman rho >= best_raw Spearman rho
     (pipeline should not be worse than the best individual proxy)
  3. full_pipeline competitive rho (GT > 50%) is clearly positive

If full_pipeline is WORSE than best_raw, this is still a valid finding --
it means the debiasing/whitening step hurts ranking on this benchmark,
which is itself an interesting result for the thesis.

---

## 11. Key Output Files to Copy Back / Keep

After the full run, the files needed for the thesis are:

```
results/nasbench101/audit/audit_summary.json
results/nasbench101/proxy_validation/correlation_results.json
results/nasbench101/debiased_proxy/partial_correlations.json
results/nasbench101/pca_whitening/pca_summary.json
results/nasbench101/surrogate_mlp/ablation_table.json
results/nasbench101/surrogate_mlp/mlp_results.json
```

These are small JSON files (< 50 KB each). They are NOT gitignored and can be
committed and pushed to git for cross-machine sync.

---

## 12. Expected Audit Numbers (Step 0 reference values from home-PC feasibility check)

These were confirmed on the home PC from the same tfrecord file.
After Step 0 completes on the lab machine, audit_summary.json should match:

| Metric | Expected Value |
|--------|----------------|
| n_total | 423,624 |
| n_missing_seeds | 0 |
| n_degenerate_lt20 | 646 |
| gt_mean | 89.68% |
| gt_std | 5.80% |
| gt_min | 9.98% |
| gt_max | 94.32% |
| gt_skewness | -8.07 (heavy left tail from degenerate archs) |
| R^2 (log_param -> GT) | 0.0471 |
| Spearman rho (param, GT) | 0.435 |
| top-1% std | 0.12% (very tight at the top -- use top-5/10% precision) |
| TFRecord load time | ~390 seconds |

If audit numbers differ significantly from the above, stop and investigate
before running proxy scripts.

Adjacency shape distribution (sample of first 5000):
  (7,7): 4221  (~84%)
  (6,6):  740  (~15%)
  (5,5):   37  (~0.7%)
  (4,4):    2  (~0.04%)

---

## 13. Comparison Table Context (for thesis writing)

These are the numbers from the two completed benchmarks.
NAS-Bench-101 results will fill in the last column.

| Metric | NAS-Bench-201 | NATS-Bench SSS | NAS-Bench-101 |
|--------|---------------|----------------|----------------|
| N architectures | 15,625 | 32,768 | 423,624 |
| Search space | Op search | Width search | DAG search |
| R^2 (size->GT) | 0.157 | 0.789 | **0.047** |
| GT range | ~10-94% | 79-94% | ~10-94% |
| GT std | ~12% | 1.27% | 5.80% |
| SynFlow partial rho | -0.002 (OUT) | +0.625 (KEEP) | ? |
| NASWOT partial rho | +0.154 (KEEP) | +0.059 (OUT) | ? |
| ZenScore partial rho | +0.190 (KEEP) | +0.051 (OUT) | ? |
| PCA input features | 3 | 2 | ? |
| full_pipeline rho | ? (Step 7 pending) | ? (Step 7 pending) | ? |

---

## 14. Quick Troubleshooting

**"ModuleNotFoundError: No module named 'proxy_utils_101'"**
  The gpu/ scripts do: sys.path.insert(0, str(Path(__file__).parent.parent))
  This adds scripts/nasbench101/ to the path. Run from the repo root.
  If it still fails: cd to repo root first.

**"CUDA device not detected"**
  Check nvidia-smi and confirm PyTorch was installed with the CUDA build.
  The scripts fall back to CPU if CUDA is unavailable -- they will still work
  but will be much slower.

**"AssertionError: adjacency must be square"**
  This should NOT happen -- the shape bug is fixed. If it does, the arch_specs.pkl
  may have been loaded from a corrupted intermediate state. Delete arch_specs.pkl
  and re-run step0_audit_101.py (this takes 390 seconds).

**OOM error on GPU during proxy compute**
  C_BASE=16 (proxy network base channels) is already very small.
  If OOM occurs, edit the gpu script to set BATCH_SIZE or INPUT_SIZE smaller.
  For NASWOT: reduce INPUT_SIZE from (8,3,32,32) to (4,3,32,32).
  For ZenScore: reduce INPUT_SIZE from (4,3,32,32) to (2,3,32,32).

**Checkpoint corruption (JSON parse error on resume)**
  Delete the checkpoint file: results/nasbench101/raw_proxy_scores/{proxy}_checkpoint.json
  The script will restart from idx=0. Progress before the corruption is lost.
  The .npy output file is only written at completion, not during checkpointing.

**protobuf error when loading TFRecord**
  Ensure PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python is set.
  Every script that imports nasbench sets this at the top. If running interactively:
    $env:PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION = "python"

---

## 15. Files That Must NOT Be Re-run After First Successful Completion

| Script | Reason |
|--------|--------|
| step0_audit_101.py | Takes ~390 seconds to load TFRecord; re-running wastes time but is safe |
| compute_proxy_param_count_101.py | Trivially fast but no need to re-run once param_count.npy exists |

All scripts are deterministic -- accidental re-runs produce identical output.
