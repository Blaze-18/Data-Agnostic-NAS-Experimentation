# NAS-Bench-101 Experimentation: Execution Strategy & Overview

**Created**: May 8, 2026  
**Document Purpose**: High-level execution strategy and decision framework for NAS-Bench-101 pipeline  
**Audience**: You (executioner) and thesis committee (reviewers)

---

## 🎯 Thesis Strategy: Three-Benchmark Validation

### The Central Question
**Can zero-cost proxies rank architectures fairly after removing structural size bias?**

### Benchmark Progression Tests

| Benchmark | R² (size→GT) | What It Tests | Current Status |
|-----------|------------|---------|---|
| **NAS-Bench-201** | 0.157 | Can proxies work when size is moderately relevant? | ✅ Steps 0-6 DONE |
| **NATS-Bench SSS** | 0.789 | Can proxies work when size DOMINATES? (stress test) | ✅ Steps 0-6 DONE |
| **NAS-Bench-101** | **0.047** | Can proxies work when size is nearly irrelevant? (LITMUS TEST) | ❌ PENDING |

### Why NAS-Bench-101 Is Critical

With R² = 0.047, parameter count explains almost **zero** variance in performance. This is the perfect stress test:

- **If proxies survive here**, they measure genuine architecture quality (not size)
- **If proxies fail here**, the methodology has fundamental limits
- **Most credible result** for the thesis (hardest test = strongest evidence)

---

## 🔄 Complete Execution Workflow

### Pre-Execution (Before Starting Phase 1)

```bash
# 1. Verify prerequisites
ls -lh data/nasbench101/nasbench_full.tfrecord  # Should be 2.1 GB
nvidia-smi                                       # Should show GTX 5070 Ti
python -c "import torch; print(torch.cuda.is_available())"  # Should print True

# 2. Set environment
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1  # Real-time output

# 3. Verify NasBench patches applied
grep -n "tf.compat.v1.io.tf_record_iterator" $(python -c "import nasbench; print(nasbench.__file__)")[:-1]/api.py
```

---

### Phase 1: Data Loading & Audit (~7 min)

**Objective**: Load TFRecord, extract all ground truth and parameter counts, verify integrity

```bash
# Single command — takes ~390 seconds
python scripts/nasbench101/step0_audit_101.py
python scripts/nasbench101/compute_proxy_param_count_101.py

# Verify outputs exist
ls -lh results/nasbench101/audit/
ls -lh results/nasbench101/raw_proxy_scores/param_count.npy
```

**Expected Output Checks**:
```python
# Check audit_summary.json
import json
with open('results/nasbench101/audit/audit_summary.json') as f:
    summary = json.load(f)
    assert summary['n_total'] == 423624, "Wrong number of architectures"
    assert summary['n_degenerate_lt20'] == 646, "Degenerate cluster mismatch"
    assert abs(summary['R2_param_count'] - 0.047) < 0.01, "R² out of range"
```

**Decision Gate**: 
- ✅ All assertions pass → proceed to Phase 2
- ❌ Any fail → STOP and debug (likely TFRecord/data issue)

---

### Phase 2: Proxy Computation (~60-90 min, parallelizable)

**Objective**: Compute 4 zero-cost proxies on all 423,624 architectures using GPU

**Execution Strategy**: Open 4 terminals

```bash
# Terminal 1: Monitor GPU
watch -n 1 nvidia-smi

# Terminal 2: SynFlow (fastest)
cd /path/to/repo
source venv/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python scripts/nasbench101/gpu/compute_proxy_synflow_gpu.py
# Expected: finishes first (~15-25 min)

# Terminal 3: NASWOT (medium speed)
cd /path/to/repo
source venv/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python scripts/nasbench101/gpu/compute_proxy_naswot_gpu.py
# Expected: finishes 2nd (~20-30 min)

# Terminal 4: ZenScore (slowest, 4x sampling)
cd /path/to/repo
source venv/bin/activate
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
python scripts/nasbench101/gpu/compute_proxy_zenscore_gpu.py
# Expected: finishes last (~25-40 min)
```

**Live Monitoring**:
- GPU usage should be ~90-95%
- Memory use ~8-10 GB (safe margin on GTX 5070 Ti)
- One complete proxy output ~400-500 MB
- Progress printed every 5,000 architectures

**Checkpoint Safety**:
- Each script checkpoints to JSON every 1,000 architectures
- If interrupted, re-run same command: auto-resumes
- No data loss (final .npy only written at completion)

**Completion Gate**: All 3 proxies complete (watch for final messages in terminals)
```bash
# Verify all three proxy files exist
ls -lh results/nasbench101/raw_proxy_scores/{synflow,naswot,zenscore}.npy
```

---

### Phase 3: Analysis & Training Pipeline (~30-40 min, sequential)

**Objective**: Transform proxies → analyze → validate → disentangle bias → whiten → train MLP

```bash
# Step 2: Transform (auto-detect negation)
python scripts/nasbench101/transform_proxies_101.py

# Step 3: Analyze distributions
python scripts/nasbench101/analyze_distributions_101.py

# Step 4: Validate ranking correlations
python scripts/nasbench101/validate_ranking_correlations_101.py
# **LOOK AT OUTPUT**: Check correlation_results.json for raw proxy ρ values
# This tells us which proxies have signal before debiasing

# Step 5: Bias disentanglement — CRITICAL DECISION GATE
python scripts/nasbench101/bias_disentanglement_101.py

# **CRITICAL**: Read partial_correlations.json before proceeding
echo "=== PROXY SURVIVAL DECISIONS ===" 
python -c "
import json
with open('results/nasbench101/debiased_proxy/partial_correlations.json') as f:
    data = json.load(f)
    for proxy in ['synflow', 'naswot', 'zenscore']:
        partial_rho = data[proxy]['partial_rank_rho']
        decision = 'KEEP' if partial_rho >= 0.10 else 'EXCLUDE'
        print(f'{proxy:12} partial_rho={partial_rho:+.4f}  {decision}')
"

# If at least 1 proxy has partial_rho >= 0.10, continue
# If all < 0.10, STOP and re-evaluate methodology (but continue anyway for thesis)

# Step 6: PCA whitening
python scripts/nasbench101/pca_whitening_101.py

# Verify whitened features are valid input for MLP
ls -lh results/nasbench101/pca_whitening/whitened_features.npy
python -c "
import numpy as np
features = np.load('results/nasbench101/pca_whitening/whitened_features.npy')
print(f'Shape: {features.shape}')
print(f'Dtype: {features.dtype}')
print(f'All finite: {np.all(np.isfinite(features))}')
print(f'Column means (should ≈ 0): {np.mean(features, axis=0)}')
print(f'Column stds (should ≈ 1): {np.std(features, axis=0)}')
"

# Step 7: Train surrogate MLP (GPU, ~15-20 min)
python scripts/nasbench101/train_mlp_101.py

# **FINAL OUTPUT**: Read ablation_table.json
echo "=== MLP PERFORMANCE COMPARISON ===" 
python -c "
import json
with open('results/nasbench101/surrogate_mlp/ablation_table.json') as f:
    data = json.load(f)
    for variant in ['size_only', 'best_raw', 'pca_raw', 'full_pipeline']:
        rho = data[variant]['test_spearman_rho']
        print(f'{variant:15} ρ = {rho:.4f}')
"
```

---

## ✅ Success Criteria & Decision Gates

### Gate 1 (After Phase 1)
**Condition**: audit_summary.json matches reference values  
**Action if PASS**: Proceed to Phase 2  
**Action if FAIL**: Debug TFRecord / NasBench patches

### Gate 2 (After Phase 2)
**Condition**: All 3 proxy .npy files exist and are valid  
**Action if PASS**: Proceed to Phase 3  
**Action if FAIL**: Resume from checkpoint or restart GPU compute

### Gate 3 (After Step 5)
**Condition**: At least 1 proxy with partial ρ ≥ 0.10  
**Action if PASS**: Continue (normal progression)  
**Action if FAIL**: Continue anyway (negative finding is still valuable), but flag in thesis

### Gate 4 (After Step 7)
**Condition**: full_pipeline ρ ≥ best_raw ρ  
**Action if PASS**: Thesis validation successful; strong methodology  
**Action if FAIL**: Methodology problematic on this benchmark; highlight in discussion

---

## 📊 Expected Outcomes & Thesis Narrative

### Most Likely Outcome (Moderate Success)

**Proxy Survival**:
- SynFlow: Likely survives (DAGs have path-capacity variation unlike SSS)
- NASWOT/ZenScore: May survive better than SSS (DAGs have op diversity)
- Predicted full_pipeline ρ: 0.30-0.45 (weak-moderate)

**Thesis Narrative**:
> "Optimal proxy selection is benchmark-dependent. On NAS-Bench-101 (DAG space with low size bias), proxies must measure genuine architecture quality. Our framework successfully disentangles size bias and produces balanced feature embeddings, enabling the surrogate to achieve moderate ranking correlation while maintaining data-agnosticism."

**Comparison Table Insight**:
```
Benchmark    | R²     | Top Proxy | Pipeline Proxy Survival
NAS-201      | 0.157  | NASWOT    | NASWOT, ZenScore survive; SynFlow fails
SSS          | 0.789  | SynFlow   | SynFlow survives; NASWOT/ZenScore fail
NAS-101      | 0.047  | ???       | NEW DATA — determines proxy universality
```

### Best Case Outcome (Strong Validation)

**Proxy Survival**: All 3 survive with complementary partial ρ signals  
**full_pipeline ρ**: Exceeds best_raw by +0.05-0.10  
**Thesis Impact**: Strongest possible validation of methodology

### Worst Case Outcome (Still Publishable)

**Proxy Survival**: None survive (all partial ρ < 0.10)  
**Interpretation**: Proxies insufficient for DAG spaces with very low size bias  
**Thesis Impact**: Negative finding; valuable contribution on methodology limitations

---

## 🚨 Contingency Protocols

### If Phase 2 GPU Script Crashes Mid-Execution

```bash
# Check checkpoint
ls -lh results/nasbench101/raw_proxy_scores/{proxy}_checkpoint.json

# Resume (will print "Resumed from checkpoint at idx=XXXX")
python scripts/nasbench101/gpu/compute_proxy_{proxy}_gpu.py

# If checkpoint corrupted:
rm results/nasbench101/raw_proxy_scores/{proxy}_checkpoint.json
python scripts/nasbench101/gpu/compute_proxy_{proxy}_gpu.py  # Restart from idx=0
```

### If Phase 3 Step Fails (Any step 2-7)

```bash
# Re-run that single step:
python scripts/nasbench101/{step_script}.py

# If still fails, check dependencies:
# - Do required input files exist? (check previous step outputs)
# - Are input shapes correct? (print shapes from .npy files)
# - Any NaN/Inf? (run validation check)
```

### If Final MLP Training Diverges (Loss = NaN or ∞)

```bash
# This shouldn't happen, but if it does:
# 1. Check whitened_features.npy is valid (no NaN/Inf)
# 2. Check target labels (GT accuracies) are valid
# 3. Try reducing learning rate in train_mlp_101.py
# 4. Try changing loss function (MSE instead of ranking loss)
```

---

## 📈 Live Dashboard Command

Run this in a separate terminal to monitor progress in real-time:

```bash
#!/bin/bash
while true; do
    clear
    echo "========== NAS-Bench-101 Execution Status =========="
    echo "Time: $(date)"
    echo ""
    echo "Phase 1 (Data Loading):"
    [ -f results/nasbench101/audit/audit_summary.json ] && echo "  ✅ Step 0 DONE" || echo "  ⏳ Step 0 pending"
    [ -f results/nasbench101/raw_proxy_scores/param_count.npy ] && echo "  ✅ Step 1a DONE" || echo "  ⏳ Step 1a pending"
    
    echo ""
    echo "Phase 2 (GPU Proxies):"
    [ -f results/nasbench101/raw_proxy_scores/synflow.npy ] && echo "  ✅ Step 1b DONE" || echo "  ⏳ Step 1b pending"
    [ -f results/nasbench101/raw_proxy_scores/naswot.npy ] && echo "  ✅ Step 1c DONE" || echo "  ⏳ Step 1c pending"
    [ -f results/nasbench101/raw_proxy_scores/zenscore.npy ] && echo "  ✅ Step 1d DONE" || echo "  ⏳ Step 1d pending"
    
    echo ""
    echo "Phase 3 (Analysis):"
    [ -d results/nasbench101/transformed_proxy ] && echo "  ✅ Step 2 DONE" || echo "  ⏳ Step 2 pending"
    [ -d results/nasbench101/proxy_distribution ] && echo "  ✅ Step 3 DONE" || echo "  ⏳ Step 3 pending"
    [ -d results/nasbench101/proxy_validation ] && echo "  ✅ Step 4 DONE" || echo "  ⏳ Step 4 pending"
    [ -f results/nasbench101/debiased_proxy/partial_correlations.json ] && echo "  ✅ Step 5 DONE" || echo "  ⏳ Step 5 pending"
    [ -f results/nasbench101/pca_whitening/whitened_features.npy ] && echo "  ✅ Step 6 DONE" || echo "  ⏳ Step 6 pending"
    [ -f results/nasbench101/surrogate_mlp/ablation_table.json ] && echo "  ✅ Step 7 DONE" || echo "  ⏳ Step 7 pending"
    
    echo ""
    echo "GPU Status:"
    nvidia-smi | head -n 10
    
    echo ""
    sleep 5
done
```

---

## 📝 Post-Execution Deliverables

After all steps complete, these files are the thesis results:

```
results/nasbench101/
├── audit/
│   └── audit_summary.json                  # Size bias quantification (R²=0.047)
├── proxy_validation/
│   └── correlation_results.json            # Raw proxy signals vs GT
├── debiased_proxy/
│   └── partial_correlations.json           # Debiasing effectiveness (KEY GATE)
├── pca_whitening/
│   └── pca_summary.json                    # Whitening validation (covariance=I)
└── surrogate_mlp/
    └── ablation_table.json                 # Pipeline comparison (FINAL METRIC)
```

**Copy to version control** (these are small JSON files):
```bash
git add results/nasbench101/{audit,proxy_validation,debiased_proxy,pca_whitening,surrogate_mlp}/*.json
git commit -m "NAS-Bench-101 complete: [SUMMARY OF RESULTS]"
git push origin main
```

---

## 🎬 Ready to Execute

**Pre-Flight Checklist**:
- [ ] TFRecord at `data/nasbench101/nasbench_full.tfrecord` (2.1 GB)
- [ ] GPU confirmed: GTX 5070 Ti available
- [ ] Python venv activated
- [ ] NasBench patches applied
- [ ] PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python set
- [ ] Results directory writable
- [ ] This document reviewed and understood

**When Ready**: Execute Phase 1, Phase 2, Phase 3 in sequence. Expected completion: ~3 hours.

**Expected Outcome**: Three-benchmark comparison table complete; proxy reversal pattern validated; thesis ready for final chapter write-up.
