# Data Processing Outputs

This folder contains outputs from the **two-stage preprocessing pipeline**:
1. **`scripts/preprocess_locally.py`** — Extract proxies, validate, transform
2. **`scripts/visualize_preprocessing.py`** — Create verification figures

## Generated Files

### Preprocessed Data (inputs for Colab training)
- **`features.npy`** — N × d float32 array of decorrelated PCA-whitened embeddings
- **`labels.npy`** — N-length float32 array of ground-truth accuracies  
- **`transforms.pkl`** — Transformation parameters (bias_coefs, scaler, imputer settings)
- **`pca_components.npy`** — PCA component matrix (d × k)
- **`pca_explained_variance.npy`** — PCA explained variance ratios

### Diagnostics
- **`validation_log.json`** — Preprocessing diagnostics (missing rates, validation failures, counts)
- **`results/preprocessing/metrics.json`** — Correlation metrics and baseline stats
- **`results/preprocessing/*.png`** — 7 verification figures (see below)

## Workflow

### Step 1: Run Preprocessing (Stage 1)
```powershell
& "F:\Thesis\Experimentation\envs\nasbench_env\Scripts\Activate.ps1"
cd F:\Thesis\Experimentation
python scripts/preprocess_locally.py
```

**Outputs:**
- `data/processed/{features, labels, transforms, validation_log}.*`
- `data/processed/{pca_components, pca_explained_variance}.npy`

**Expected time:** 30–60 min (depends on disk I/O and imputation quality)

### Step 2: Run Visualization (Stage 2)
```powershell
python scripts/visualize_preprocessing.py
```

**Outputs:**
- `results/preprocessing/metrics.json` — Correlation and baseline scores
- `results/preprocessing/*.png` — Verification figures:
  1. `01_pca_variance.png` — PCA explained variance per component
  2. `02_pca0_vs_label.png` — Scatter plot: first PCA dim vs label
  3. `03_linear_pred_vs_true.png` — Linear baseline predictions
  4. `04_pca_correlation.png` — Correlation heatmap of PCA components
  5. `05_label_distribution.png` — Histogram of ground-truth labels
  6. `06_pca_label_correlations.png` — Bar: Spearman ρ per PCA component
  7. `07_validation_summary.png` — Text summary of preprocessing results

## Validation Checklist

✓ **All required files exist?**  
  `features.npy`, `labels.npy`, `transforms.pkl`, `pca_components.npy`

✓ **Proxy signal present?**  
  Check `results/preprocessing/metrics.json` → `correlations.linear_on_pca.spearman` > 0.3 ✓

✓ **Label distribution reasonable?**  
  Review `05_label_distribution.png` for shape/range (should be ~0.5–1.0 or 50–100%)

✓ **PCA components decorrelated?**  
  `04_pca_correlation.png` should show near-identity (diagonal only)

✓ **No major failure flags?**  
  Check `validation_log.json` for failed loads or validation failures

## Key Parameters

| Parameter | Value |
|-----------|-------|
| Bias disentanglement | Linear regression: residuals of proxies after regressing out size |
| Imputation | Median of non-NaN values per column |
| Scaling | StandardScaler (zero-mean, unit variance) |
| PCA | Full components, whitened=True |
| Output dtype | float32 (memory efficient) |

## Troubleshooting

**Issue: "No .pth files found"**  
→ Ensure `data/chunks_clean/arch2infos/` exists and is populated

**Issue: "No valid records extracted"**  
→ Check `validation_log.json` for failure reasons (e.g., all_proxies_nan, label_out_of_range)

**Issue: Very low Spearman ρ (<0.1)**  
→ Proxies may not have signal; check if chunk files embed proxy fields or if labels are corrupt

**Issue: Missing imputer_statistics in transforms.pkl**  
→ This is expected if imputation is skipped; safe to ignore

## Next Steps

1. Review validation figures in `results/preprocessing/`
2. Check `metrics.json` — Spearman ρ should be > 0.3 for meaningful signal  
3. Upload `data/processed/` folder to Google Drive
4. Create + run `notebooks/colab_training.ipynb` to train MLP on GPU
