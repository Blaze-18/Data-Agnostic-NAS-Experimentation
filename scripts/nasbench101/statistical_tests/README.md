# Statistical Validation Scripts — NAS-Bench-101

These three scripts convert the Step 7 ablation point estimates into
statistically verified, reproducible findings.

## Folder Structure

```
scripts/nasbench101/statistical_tests/
    phase1_save_predictions.py   ← prerequisite for phases 2 and 3 (Test 1 & 3)
    phase2_statistical_tests.py  ← Test 1 (Bootstrap CI) + Test 3 (Competitive subset)
    phase3_multi_seed.py         ← Test 2 (Multi-seed stability)

results/nasbench101/statistical_validation/
    predictions/
        test_pred_pca_raw.npy
        test_pred_full_pipeline.npy
        test_gt.npy
        test_indices.npy
    bootstrap_results.json
    bootstrap_delta_histogram.png
    competitive_subset_analysis.json
    competitive_subset_plot.png
    phase2_summary.json
    multi_seed_results.json
    multi_seed_gap_plot.png
```

## Execution Order

### Phase 1 — Save test-set predictions (~5 min GPU)
```bash
python scripts/nasbench101/statistical_tests/phase1_save_predictions.py
```
Retrains pca_raw and full_pipeline with the canonical seed-42 split and saves
predictions on the held-out test set. Required before Phase 2.

### Phase 2 — Tests 1 and 3 (~15 min GPU + CPU)
```bash
python scripts/nasbench101/statistical_tests/phase2_statistical_tests.py
```
**Test 1 — Bootstrap CI**: 1000 resamples of the test set. Reports 95% CI
of Δρ = ρ(full_pipeline) − ρ(pca_raw). If CI excludes zero, the debiasing
contribution is statistically significant.

**Test 3 — Competitive subset**: Spearman ρ for all 4 variants at three
GT accuracy thresholds (>50%, >85%, >90%). Shows whether the debiasing
advantage holds as the ranking task becomes harder.

### Phase 3 — Test 2 (~2 hours GPU, run overnight)
```bash
python scripts/nasbench101/statistical_tests/phase3_multi_seed.py
```
Trains pca_raw and full_pipeline across 5 seeds (0, 1, 2, 3, 42).
Seed 42 is loaded from the existing mlp_results.json without retraining.
Reports per-seed ρ table, mean ± std gap, and a gap stability plot.

To run only specific seeds (e.g., the 4 new ones):
```bash
python scripts/nasbench101/statistical_tests/phase3_multi_seed.py --seeds 0 1 2 3
```

## What Each Test Proves

| Test | Claim it validates |
|------|--------------------|
| **Test 1** (Bootstrap CI) | The +0.022 Δρ gap is not sampling noise — statistically significant |
| **Test 2** (Multi-seed) | The gap is reproducible across different data splits — not a seed artifact |
| **Test 3** (Competitive subset) | The gap holds in the architecturally meaningful regime (GT > 85%, GT > 90%) |
