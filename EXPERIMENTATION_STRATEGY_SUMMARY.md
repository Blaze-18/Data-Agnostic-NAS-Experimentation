# Complete Experimentation Strategy Summary

**Date Created**: May 8, 2026  
**Project Status**: Ready for NAS-Bench-101 execution  
**Documentation**: Comprehensive plan created for immediate execution

---

## 📚 Documents Created

### 1. **Experimentation_Progress_NAS101.md** (584 lines)
- **Purpose**: Detailed step-by-step breakdown of all 7 steps
- **Content**: 
  - Phase 1 (Step 0-1a): Data loading & audit details
  - Phase 2 (Step 1b-1d): GPU proxy computation specifics
  - Phase 3 (Step 2-7): Full analysis & training pipeline
  - Risk assessment & contingency plans
  - Success metrics & decision gates
  - Monitoring checkpoints
- **Use**: Reference during execution; update progress log in real-time
- **Audience**: You (during execution) and thesis committee (reviewing approach)

### 2. **EXECUTION_STRATEGY.md** (350+ lines)
- **Purpose**: Quick reference execution guide (not exhaustive, but complete)
- **Content**:
  - Thesis strategy & why NAS-101 is critical
  - Complete workflow with bash commands
  - Success criteria & decision gates
  - Contingency protocols
  - Live dashboard monitoring script
  - Post-execution deliverables checklist
- **Use**: Copy-paste commands; follow if-then decisions
- **Audience**: You (during active execution); technical reference

### 3. **README.md** (Updated)
- **Purpose**: Overall project documentation
- **Changes**: Rewrote to reflect current status
- **Content**: 3-benchmark overview, current status, pipeline steps, key files
- **Use**: High-level reference; shared with collaborators
- **Audience**: General audience; thesis committee

---

## 🎯 Three-Phase Execution Overview

### Phase 1: Data Loading & Audit (7 min)
```
Step 0: Load TFRecord                  →  6.5 min (390s load time)
Step 1a: Extract param_count           →  <1 min
Expected: n_total=423,624, R²≈0.047    →  ✅ GATE: Check audit_summary.json
```

### Phase 2: GPU Proxy Computation (45-90 min, parallelizable)
```
Step 1b: SynFlow (GPU)                 →  15-25 min  |
Step 1c: NASWOT (GPU)                  →  20-30 min  | Run in parallel
Step 1d: ZenScore (GPU)                →  25-40 min  |
Expected: 3 × 423K proxy scores        →  ✅ GATE: All .npy files exist
```

### Phase 3: Analysis & Training (30-40 min, sequential)
```
Step 2: Log-transform                  →  1 min
Step 3: Analyze distributions          →  1 min
Step 4: Validate ranking correlations  →  3 min   (observe raw ρ)
Step 5: Bias disentanglement           →  5 min   ⚠️ CRITICAL GATE
Step 6: PCA whitening                  →  1 min
Step 7: MLP training                   →  15-20 min (GPU, pairwise ranking loss)
Expected: full_pipeline ρ result       →  ✅ GATE: Check ablation_table.json
```

**Total Time**: ~2-3 hours start to finish

---

## ⚠️ Critical Decision Gates

### Gate 1: After Phase 1
**Check**: `results/nasbench101/audit/audit_summary.json`
```json
{
  "n_total": 423624,
  "n_degenerate_lt20": 646,
  "gt_mean": 89.68,
  "gt_std": 5.80,
  "R2_param_count": 0.047
}
```
- ✅ **PASS**: All within expected ranges → Continue to Phase 2
- ❌ **FAIL**: Mismatch → Stop, debug TFRecord/patches

### Gate 2: After Phase 2
**Check**: File existence & size
```bash
ls -lh results/nasbench101/raw_proxy_scores/{synflow,naswot,zenscore}.npy
# Each should be ~400-500 MB, dtype float64
```
- ✅ **PASS**: All three files exist → Continue to Phase 3
- ❌ **FAIL**: Missing → Resume from checkpoint or restart

### Gate 3: After Step 5 (MOST CRITICAL)
**Check**: `results/nasbench101/debiased_proxy/partial_correlations.json`
```json
{
  "synflow": {"partial_rank_rho": ???, "decision": "KEEP/EXCLUDE"},
  "naswot": {"partial_rank_rho": ???, "decision": "KEEP/EXCLUDE"},
  "zenscore": {"partial_rank_rho": ???, "decision": "KEEP/EXCLUDE"}
}
```
- ✅ **PASS**: At least 1 proxy with partial_rho ≥ 0.10 → Continue
- ⚠️ **CONDITIONAL PASS**: All proxies < 0.10 → Continue anyway (methodology test), but flag in thesis
- ❌ **FAIL**: Should not happen; if it does, re-run Step 5

### Gate 4: After Step 7 (FINAL)
**Check**: `results/nasbench101/surrogate_mlp/ablation_table.json`
```json
{
  "size_only": {"test_spearman_rho": ???},
  "best_raw": {"test_spearman_rho": ???},
  "pca_raw": {"test_spearman_rho": ???},
  "full_pipeline": {"test_spearman_rho": ???}
}
```
- ✅ **STRONG**: full_pipeline ≥ best_raw + 0.05 → Excellent methodology validation
- ✅ **ACCEPTABLE**: full_pipeline ≈ best_raw (within 0.05) → Debiasing neutral, not harmful
- ⚠️ **CONCERNING**: full_pipeline < best_raw by >0.10 → Methodology problematic on this benchmark (but still publish)
- ❌ **FAILURE**: full_pipeline < 0.10 → No signal at all (rare)

---

## 🔄 Proxy Survival Prediction

Based on two completed benchmarks, NAS-Bench-101 may show:

### Scenario A: All Three Survive (Optimal)
- **SynFlow**: partial ρ ≥ 0.15 (DAGs have path-capacity variation unlike SSS)
- **NASWOT**: partial ρ ≥ 0.15 (DAGs have operation diversity unlike SSS)
- **ZenScore**: partial ρ ≥ 0.15 (same as NASWOT)
- **Thesis Impact**: Strongest validation of proxy reversal pattern
- **Expected full_pipeline ρ**: 0.35-0.45

### Scenario B: Two Survive (Likely)
- **SynFlow + NASWOT** or **SynFlow + ZenScore** or **NASWOT + ZenScore**
- **Thesis Impact**: Validates benchmark-dependence of proxy selection
- **Expected full_pipeline ρ**: 0.25-0.35

### Scenario C: One Survives (Possible)
- **Only SynFlow** (best case): Shows path-capacity dominates DAG diversity
- **Only NASWOT/ZenScore** (unexpected): Shows op diversity dominates path-capacity
- **Thesis Impact**: Narrows down which architectural property is universal
- **Expected full_pipeline ρ**: 0.20-0.30

### Scenario D: None Survive (Stress Test)
- **All partial ρ < 0.10**: Proxies insufficient for very low-bias benchmarks
- **Thesis Impact**: Identifies methodology limitations; still publishable
- **Expected full_pipeline ρ**: <0.15 (essentially noise)

---

## 📊 Three-Benchmark Comparison (Will Be Completed After NAS-101)

```
┌─────────────────────┬──────────────┬──────────────┬──────────────┐
│ Metric              │ NAS-Bench-201│ NATS-Bench SSS│ NAS-Bench-101│
├─────────────────────┼──────────────┼──────────────┼──────────────┤
│ N architectures     │ 15,625       │ 32,768       │ 423,624      │
│ Search space        │ Op search    │ Width only   │ DAG search   │
│ R² (param→GT)       │ 0.157        │ 0.789        │ 0.047        │
│ Size relevance      │ Moderate     │ DOMINATES    │ IRRELEVANT   │
│ GT std              │ ~12%         │ 1.27%        │ ~5.8%        │
├─────────────────────┼──────────────┼──────────────┼──────────────┤
│ SynFlow partial ρ   │ -0.002 (OUT) │ +0.625 (IN)  │ ???          │
│ NASWOT partial ρ    │ +0.154 (IN)  │ +0.059 (OUT) │ ???          │
│ ZenScore partial ρ  │ +0.190 (IN)  │ +0.051 (OUT) │ ???          │
├─────────────────────┼──────────────┼──────────────┼──────────────┤
│ PCA input features  │ 3            │ 2            │ ???          │
│ full_pipeline ρ     │ Pending      │ Pending      │ TO RUN       │
└─────────────────────┴──────────────┴──────────────┴──────────────┘
```

**Key Insight**: If proxy survival differs across all three benchmarks → **Proxy selection is benchmark-dependent** (thesis claim validated)

---

## 🚀 How to Start Execution

### Step 1: Verify Prerequisites
```bash
cd /home/anan/NAS/Experimentation/Data-Agnostic-NAS-Experimentation

# Check TFRecord
ls -lh data/nasbench101/nasbench_full.tfrecord  # Should be 2.1 GB

# Check GPU
nvidia-smi | grep "Tesla\|GTX"  # Should show GTX 5070 Ti

# Check Python & CUDA
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Check environment
echo $PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION  # Should be 'python'
```

### Step 2: Execute Phase 1 (Once)
```bash
python scripts/nasbench101/step0_audit_101.py
python scripts/nasbench101/compute_proxy_param_count_101.py

# Verify
cat results/nasbench101/audit/audit_summary.json | head -20
```

### Step 3: Execute Phase 2 (Parallel, in 4 terminals)
```bash
# Terminal 1: GPU monitor
watch -n 1 nvidia-smi

# Terminal 2: SynFlow
python scripts/nasbench101/gpu/compute_proxy_synflow_gpu.py

# Terminal 3: NASWOT
python scripts/nasbench101/gpu/compute_proxy_naswot_gpu.py

# Terminal 4: ZenScore
python scripts/nasbench101/gpu/compute_proxy_zenscore_gpu.py
```

### Step 4: Execute Phase 3 (Sequential, single terminal)
```bash
# All steps in one script
for step in \
  "scripts/nasbench101/transform_proxies_101.py" \
  "scripts/nasbench101/analyze_distributions_101.py" \
  "scripts/nasbench101/validate_ranking_correlations_101.py" \
  "scripts/nasbench101/bias_disentanglement_101.py" \
  "scripts/nasbench101/pca_whitening_101.py" \
  "scripts/nasbench101/train_mlp_101.py"
do
  echo "Running: $step"
  python "$step"
  [ $? -ne 0 ] && echo "FAILED: $step" && break
done

# Check final results
cat results/nasbench101/surrogate_mlp/ablation_table.json
```

### Step 5: Verify & Document Results
```bash
# Create summary
python -c "
import json
print('\\n=== NAS-BENCH-101 RESULTS ===\\n')

with open('results/nasbench101/proxy_validation/correlation_results.json') as f:
    val = json.load(f)
    print('Raw Proxy Correlations:')
    for proxy in ['param_count', 'synflow', 'naswot', 'zenscore']:
        rho = val[proxy]['spearman_rho']
        print(f'  {proxy:12} ρ = {rho:+.4f}')

with open('results/nasbench101/debiased_proxy/partial_correlations.json') as f:
    debiased = json.load(f)
    print('\\nAfter Bias Disentanglement:')
    for proxy in ['synflow', 'naswot', 'zenscore']:
        partial = debiased[proxy]['partial_rank_rho']
        decision = debiased[proxy]['decision']
        print(f'  {proxy:12} partial_ρ = {partial:+.4f}  [{decision}]')

with open('results/nasbench101/surrogate_mlp/ablation_table.json') as f:
    ablation = json.load(f)
    print('\\nMLP Performance (Ablation):')
    for variant in ['size_only', 'best_raw', 'pca_raw', 'full_pipeline']:
        rho = ablation[variant]['test_spearman_rho']
        print(f'  {variant:15} ρ = {rho:+.4f}')
"
```

---

## 📋 Post-Execution Next Steps

After NAS-Bench-101 completes:

1. **Train MLPs for NAS-Bench-201 & SSS** (Step 7 for both)
2. **Generate final 3-benchmark comparison table**
3. **Write thesis Chapter 5: Empirical Results**
4. **Create visualization**: Proxy reversal pattern (3-panel figure)
5. **Commit all results to git**

---

## 📞 Key Contact Information

**If Issues Arise**:
- **TFRecord problems**: See `LAB_MACHINE_CONTEXT_NASBENCH101.md` section 5
- **GPU issues**: Check NVIDIA documentation; reduce batch size in scripts
- **API patches**: Verify all 3 patches applied to `nasbench/api.py`
- **Algorithm questions**: Reference `EXPERIMENTATION_PROGRESS_NOTE.md` (NAS-201) and `EXPERIMENTATION_NOTE_SSS.md`

---

## ✅ Checklist Before Starting

- [ ] Read this entire document (you are here ✓)
- [ ] Read `Experimentation_Progress_NAS101.md` (detailed reference)
- [ ] Read `EXECUTION_STRATEGY.md` (command reference)
- [ ] Verify TFRecord exists (2.1 GB)
- [ ] Verify GPU available (GTX 5070 Ti)
- [ ] Verify NasBench patches applied
- [ ] Verify environment variables set
- [ ] Create backup of code (git commit current state)
- [ ] Open 4 terminals for Phase 2 (or use tmux/screen)
- [ ] Set up monitoring dashboard (optional but recommended)

---

## 🎬 Ready to Execute

**Expected Timeline**:
- 14:00 UTC: Start Phase 1 (~7 min)
- 14:10 UTC: Start Phase 2 (~90 min max)
- 15:45 UTC: Start Phase 3 (~40 min)
- 16:30 UTC: **All results ready** → 3-benchmark table complete

**Expected Outcome**: Thesis fully validated with NAS-Bench-101 as litmus test.

---

**Status**: 🟢 **READY TO EXECUTE**
