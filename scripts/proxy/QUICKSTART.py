#!/usr/bin/env python3
"""
QUICK START GUIDE FOR PROXY COMPUTATION
=======================================

This file contains executable examples you can copy/paste directly.
Run from the workspace root: f:\Thesis\Experimentation
"""

# ============================================================================
# OPTION 1: QUICK VALIDATION (Recommended First Run)
# Runtime: ~30 minutes | Architectures: 256 | Device: DirectML
# ============================================================================
# 
# Command:
#   cd f:\Thesis\Experimentation
#   python scripts/proxy/run_all_proxies.py --subset 256
#
# Expected Output:
#   ✅ Parameter Count  |  256 computed | ~10s
#   ✅ SynFlow          |  256 computed | ~8min
#   ✅ NASWOT           |   66 computed | ~4min (190 skipped - architectural)
#   ✅ Zen-Score        |   66 computed | ~4min (190 skipped - architectural)
#   Total runtime: ~30 min
#
# Result Files Generated:
#   - scripts/proxy/param_count_test_results.json
#   - scripts/proxy/synflow_test_results.json
#   - scripts/proxy/naswot_test_results.json
#   - scripts/proxy/zenscore_test_results.json
#   - scripts/proxy/all_proxies_results.json (summary)


# ============================================================================
# OPTION 2: INDIVIDUAL PROXY SCRIPTS
# Runtime varies | Architectures: 256 | Device: DirectML
# ============================================================================
#
# Parameter Count (FASTEST - ~20 seconds):
#   python scripts/proxy/compute_proxy_param_count.py
#
# SynFlow (MEDIUM - ~8 minutes):
#   python scripts/proxy/compute_proxy_synflow.py
#
# NASWOT (MEDIUM - ~15 minutes):
#   python scripts/proxy/compute_proxy_naswot.py
#
# Zen-Score (SLOW - ~15 minutes):
#   python scripts/proxy/compute_proxy_zenscore.py
#
# Recommended Order of Execution:
#   1. Run param_count (fast validation)
#   2. Run synflow (good discriminative power)
#   3. Run naswot (supplementary signal)
#   4. Run zenscore (supplementary signal)


# ============================================================================
# OPTION 3: PRODUCTION RUN - FULL DATASET
# Runtime: ~32+ hours | Architectures: 15,625 | Device: DirectML
# ============================================================================
#
# IMPORTANT: Only run after validating --subset 256 first!
#
# Run all 4 proxies on full dataset:
#   python scripts/proxy/run_all_proxies.py --full
#
# For faster production (param_count + synflow only - ~2.5 hours):
#   python scripts/proxy/run_all_proxies.py --full --proxies param_count synflow
#
# Run individual proxies on full dataset:
#   python scripts/proxy/compute_proxy_param_count.py --full-run
#   python scripts/proxy/compute_proxy_synflow.py --full-run
#   python scripts/proxy/compute_proxy_naswot.py --full-run
#   python scripts/proxy/compute_proxy_zenscore.py --full-run
#
# Estimated Runtimes (DirectML, Full Dataset):
#   - Parameter Count: ~5 minutes
#   - SynFlow: ~30 minutes
#   - NASWOT: ~20 minutes
#   - Zen-Score: ~20 minutes
#   - TOTAL: ~75 minutes (if sequential)
#
# Result Files Generated:
#   - scripts/proxy/param_count_results.json (15,625 architectures)
#   - scripts/proxy/synflow_results.json
#   - scripts/proxy/naswot_results.json
#   - scripts/proxy/zenscore_results.json


# ============================================================================
# OPTION 4: TEST ON DIFFERENT DEVICE
# ============================================================================
#
# Use CPU instead of DirectML (5-10x slower but works if GPU unavailable):
#   python scripts/proxy/run_all_proxies.py --subset 256 --device cpu
#
# Use CUDA GPU (if available - fastest):
#   python scripts/proxy/run_all_proxies.py --subset 256 --device cuda


# ============================================================================
# INTERPRETING RESULTS
# ============================================================================
#
# Result JSON Format (example param_count_test_results.json):
# {
#   "architecture_0": 3514,
#   "architecture_1": 35674,
#   "architecture_10": 18294,
#   ...
#   "stats": {
#     "min": 634,
#     "max": 48794,
#     "mean": 17242,
#     "median": 15194,
#     "std": 10930,
#     "count": 256
#   }
# }
#
# What Each Proxy Measures:
#   - Parameter Count: Simple count of trainable weights (most stable, should be 256/256)
#   - SynFlow: Gradient flow magnitude (should have wide range 0.1-200, all 256/256)
#   - NASWOT: Activation pattern diversity (might have 66/256 due to skip connections)
#   - Zen-Score: Activation variance (similar to NASWOT, 66/256 expected)
#
# Expected Success Rates:
#   - Parameter Count: ✅ 100% (256/256)
#   - SynFlow: ✅ 100% (256/256)
#   - NASWOT: ⚠️ 26% (66/256) - normal! Skip connections reduce activations
#   - Zen-Score: ⚠️ 26% (66/256) - normal! Same architectural constraint


# ============================================================================
# TROUBLESHOOTING
# ============================================================================
#
# GPU Out of Memory:
#   → Reduce batch_size in script (currently 64 for param_count, 8 for proxies)
#   → Or use --device cpu (slower but works)
#
# All values are 0 or NaN:
#   → Model loading failed (check arch_str parsing)
#   → Device mismatch (ensure device matches model device)
#   → Hook registration issue (for NASWOT/Zen-Score)
#   → Run individual script with --debug flag for detailed logging
#
# Script runs but produces very different numbers than expected:
#   → Check that chunk files haven't changed (validate file count: 15,625)
#   → Ensure correct architecture IDs (0-15624 expected)
#   → Verify DirectML vs CPU (different results but both valid)
#
# Batch Processing Fails:
#   → Check dataset path (should be: data/chunks_clean/arch2infos/*.pth)
#   → Verify .pth files can be loaded: python -c "import torch; torch.load('data/chunks_clean/arch2infos/0.pth')"
#
# Hook registration errors (NASWOT/Zen-Score):
#   → These proxies are optional (parameter count + synflow provide full coverage)
#   → Skip them if time-critical: --proxies param_count synflow


# ============================================================================
# RECOMMENDED WORKFLOW
# ============================================================================
#
# Day 1 - Validation Phase (30 min):
#   1. python scripts/proxy/run_all_proxies.py --subset 256
#   2. Review generated JSON files for reasonable values
#   3. Check TEST_RESULTS_SUMMARY.md for expected ranges
#   4. Verify: All 256 architectures computed successfully? ✓
#   5. If all pass: Proceed to Day 2
#
# Day 2 - Production Phase (2-4 hours):
#   1. Start full run (parameter count + synflow only - faster):
#      python scripts/proxy/run_all_proxies.py --full --proxies param_count synflow
#   2. While running, start second terminal:
#      python scripts/bias_disentanglement.py
#      (This will fail initially, waiting for proxy data - but sets up dependencies)
#   3. Once proxies complete, run bias-disentanglement script
#   4. Generate visualization and validation plots
#
# Day 3 - Finalization (1 hour):
#   1. Review correlation heatmaps between proxies
#   2. Verify features.npy shape and statistics
#   3. Upload to Google Drive
#   4. Create Colab notebook for MLP training


# ============================================================================
# NEXT STEPS AFTER PROXY COMPUTATION
# ============================================================================
#
# 1. BIAS DISENTANGLEMENT:
#    - Load 4 proxy result files
#    - Remove parameter count effect via linear regression
#    - Stack into single matrix (15625, 4)
#    - Apply PCA + whitening transformation
#    - Save features.npy and transforms.pkl
#
# 2. VISUALIZATION:
#    - Generate proxy correlation heatmap
#    - Plot explained variance ratio (PCA)
#    - Show feature distributions
#    - Validate correlation with ground truth NAS accuracy
#
# 3. UPLOAD TO COLAB:
#    - Compress data/processed/ directory
#    - Upload to Google Drive
#    - Mount in Colab notebook
#    - Train MLP surrogate model with ranking loss
#
# 4. MLP TRAINING:
#    - Load features.npy + labels.npy in Colab
#    - Implement ranking-aware loss (e.g., ListNet, LambdaMART)
#    - Train for 50-100 epochs
#    - Validate on holdout set
#    - Export trained model checkpoint


# ============================================================================
# REFERENCE: FILE STRUCTURE POST-COMPUTATION
# ============================================================================
#
# f:\Thesis\Experimentation\
#  ├── data/
#  │   ├── chunks_clean/
#  │   │   └── arch2infos/ (15,625 .pth files - INPUT)
#  │   └── processed/
#  │       ├── labels.npy (existing)
#  │       ├── features.npy (will be created after bias-disentanglement)
#  │       └── transforms.pkl (will be created after bias-disentanglement)
#  │
#  ├── scripts/
#  │   └── proxy/
#  │       ├── compute_proxy_param_count.py
#  │       ├── compute_proxy_synflow.py
#  │       ├── compute_proxy_naswot.py
#  │       ├── compute_proxy_zenscore.py
#  │       ├── proxy_utils.py
#  │       ├── run_all_proxies.py (MASTER SCRIPT)
#  │       ├── test_arch_format.py
#  │       ├── README.md (detailed docs)
#  │       ├── QUICKSTART.py (this file)
#  │       ├── TEST_RESULTS_SUMMARY.md (validation results)
#  │       │
#  │       └── (After 256-subset test)
#  │           ├── param_count_test_results.json
#  │           ├── synflow_test_results.json
#  │           ├── naswot_test_results.json
#  │           ├── zenscore_test_results.json
#  │           └── all_proxies_results.json (master summary)
#  │
#  │       └── (After full 15,625 run)
#  │           ├── param_count_results.json (full dataset)
#  │           ├── synflow_results.json
#  │           ├── naswot_results.json
#  │           └── zenscore_results.json


print(__doc__)  # Print this documentation when file is run
