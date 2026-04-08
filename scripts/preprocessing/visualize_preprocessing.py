#!/usr/bin/env python3
"""
Stage 2: Load preprocessed features and generate verification visualizations.

Loads: data/processed/{features,labels,transforms}.* + validation_log.json
Saves: results/preprocessing/*.png + metrics.json

Run after preprocess_locally.py:
    python scripts/visualize_preprocessing.py
"""
import json
import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, kendalltau
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns


def main():
    indir = Path("data/processed")
    outdir = Path("results/preprocessing")
    outdir.mkdir(parents=True, exist_ok=True)

    # Validate outputs exist
    required = ["features.npy", "labels.npy", "transforms.pkl", "pca_components.npy", "pca_explained_variance.npy"]
    for f in required:
        if not (indir / f).exists():
            print(f"ERROR: Missing {f} in {indir}. Run preprocess_locally.py first.")
            return

    print(f"Loading preprocessed data from {indir}...")
    features = np.load(indir / "features.npy")
    labels = np.load(indir / "labels.npy")
    pca_components = np.load(indir / "pca_components.npy")
    pca_explained_var = np.load(indir / "pca_explained_variance.npy")

    with open(indir / "transforms.pkl", "rb") as f:
        transforms = pickle.load(f)

    with open(indir / "validation_log.json", "r") as f:
        val_log = json.load(f)

    proxy_cols = transforms["proxy_cols"]
    size_cols = transforms["size_cols"]

    print(f"Features shape: {features.shape}, Labels shape: {labels.shape}")

    # ========== VERIFY & COMPUTE METRICS ==========
    metrics = {}

    # Basic stats
    metrics["features"] = {
        "shape": features.shape,
        "dtype": str(features.dtype),
        "mean": [float(features[:, i].mean()) for i in range(features.shape[1])],
        "std": [float(features[:, i].std()) for i in range(features.shape[1])],
    }
    metrics["labels"] = {
        "shape": labels.shape,
        "min": float(labels.min()),
        "max": float(labels.max()),
        "mean": float(labels.mean()),
        "std": float(labels.std()),
    }

    # PCA variance
    metrics["pca"] = {
        "n_components": len(pca_explained_var),
        "explained_variance": pca_explained_var.tolist(),
        "cumsum_variance": float(pca_explained_var.sum()),
    }

    # Correlations: PCA dims vs labels
    metrics["correlations"] = {}
    for i in range(features.shape[1]):
        rho, _ = spearmanr(features[:, i], labels)
        tau, _ = kendalltau(features[:, i], labels)
        metrics["correlations"][f"pca_{i}"] = {
            "spearman": float(rho) if not np.isnan(rho) else None,
            "kendall": float(tau) if not np.isnan(tau) else None,
        }

    # Linear baseline: fit LR from features -> labels
    lr = LinearRegression()
    lr.fit(features, labels)
    preds = lr.predict(features)
    rho, _ = spearmanr(preds, labels)
    tau, _ = kendalltau(preds, labels)
    metrics["correlations"]["linear_on_pca"] = {
        "spearman": float(rho) if not np.isnan(rho) else None,
        "kendall": float(tau) if not np.isnan(tau) else None,
    }

    # Bias coefficients (quality ofregression for each proxy)
    metrics["bias_disentanglement"] = {}
    for pc, coef_dict in transforms["bias_coefs"].items():
        metrics["bias_disentanglement"][pc] = {
            "r2_score": coef_dict.get("r2", None),
            "intercept": coef_dict["intercept"],
        }

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics.json")

    # ========== VISUALIZATIONS ==========
    print("Generating visualizations...")
    
    # 1. PCA explained variance
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(pca_explained_var)), pca_explained_var)
        ax.set_xlabel("PCA Component")
        ax.set_ylabel("Explained Variance Ratio")
        ax.set_title("PCA Explained Variance (Whitened)")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "01_pca_variance.png", dpi=150)
        plt.close()
        print("  ✓ 01_pca_variance.png")
    except Exception as e:
        print(f"  ✗ PCA variance plot failed: {e}")

    # 2. First PCA component vs label
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(features[:, 0], labels, alpha=0.4, s=20)
        ax.set_xlabel("PCA Component 0")
        ax.set_ylabel("Label (Accuracy)")
        ax.set_title("First PCA Component vs Label")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "02_pca0_vs_label.png", dpi=150)
        plt.close()
        print("  ✓ 02_pca0_vs_label.png")
    except Exception as e:
        print(f"  ✗ PCA0 vs label plot failed: {e}")

    # 3. Linear fit predictions vs true
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(labels, preds, alpha=0.4, s=20)
        lims = [min(labels.min(), preds.min()), max(labels.max(), preds.max())]
        ax.plot(lims, lims, 'r--', lw=2, label='Perfect prediction')
        ax.set_xlabel("True Label")
        ax.set_ylabel("Linear Model Prediction")
        ax.set_title("Linear Baseline: Predictions vs True")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "03_linear_pred_vs_true.png", dpi=150)
        plt.close()
        print("  ✓ 03_linear_pred_vs_true.png")
    except Exception as e:
        print(f"  ✗ Linear prediction plot failed: {e}")

    # 4. Correlation heatmap: PCA components
    try:
        corr_matrix = np.corrcoef(features.T)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                    square=True, ax=ax, cbar_kws={"label": "Correlation"})
        ax.set_title("Correlation: PCA Components (Residuals)")
        plt.tight_layout()
        plt.savefig(outdir / "04_pca_correlation.png", dpi=150)
        plt.close()
        print("  ✓ 04_pca_correlation.png")
    except Exception as e:
        print(f"  ✗ Correlation heatmap failed: {e}")

    # 5. Label distribution
    try:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(labels, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel("Label (Accuracy)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Ground-Truth Labels")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "05_label_distribution.png", dpi=150)
        plt.close()
        print("  ✓ 05_label_distribution.png")
    except Exception as e:
        print(f"  ✗ Label distribution plot failed: {e}")

    # 6. Correlation of each PCA component with label
    try:
        corrs = []
        for i in range(features.shape[1]):
            rho, _ = spearmanr(features[:, i], labels)
            corrs.append(rho if not np.isnan(rho) else 0)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(range(len(corrs)), corrs, color=['green' if c > 0 else 'red' for c in corrs])
        ax.set_xlabel("PCA Component")
        ax.set_ylabel("Spearman Correlation with Label")
        ax.set_title("Correlation: Each PCA Dimension vs Label")
        ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(outdir / "06_pca_label_correlations.png", dpi=150)
        plt.close()
        print("  ✓ 06_pca_label_correlations.png")
    except Exception as e:
        print(f"  ✗ PCA-label correlation plot failed: {e}")

    # 7. Validation summary (text figure)
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        summary_text = f"""
PREPROCESSING VALIDATION SUMMARY

Data Loading:
  Total files attempted: {val_log.get('total_files', 'N/A')}
  Successfully loaded: {val_log.get('loaded_successfully', 'N/A')}
  Load failures: {val_log.get('load_failed', 'N/A')}
  
Records:
  Before validation: {val_log.get('records_before_validation', 'N/A')}
  After validation: {val_log.get('records_after_validation', 'N/A')}
  With proxy fields: {val_log.get('has_proxy_fields_count', 'N/A')}
  
Validation Failures: {val_log.get('validation_failures', {})}
  
Features:
  Shape: {features.shape}
  Mean std: {np.mean(features.std(axis=0)):.4f}
  
Labels:
  Min: {labels.min():.4f}
  Max: {labels.max():.4f}
  Mean: {labels.mean():.4f}
  Std: {labels.std():.4f}
  
Linear Baseline (PCA → Label):
  Spearman ρ: {metrics['correlations']['linear_on_pca']['spearman']:.4f}
  Kendall τ: {metrics['correlations']['linear_on_pca']['kendall']:.4f}
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
                fontfamily='monospace', fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.savefig(outdir / "07_validation_summary.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  ✓ 07_validation_summary.png")
    except Exception as e:
        print(f"  ✗ Validation summary plot failed: {e}")

    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Figures saved to: {outdir}/")
    print(f"Metrics saved to: {outdir}/metrics.json")
    print("\nRecommendation:")
    print("  1. Review the figures in results/preprocessing/")
    print("  2. Check metrics.json for correlation values")
    print("  3. If Spearman ρ (linear baseline) is > 0.3, proxies carry signal ✓")
    print("  4. Upload data/processed/ to Google Drive")
    print("  5. Run Colab training notebook")
    print("="*60)


if __name__ == "__main__":
    main()
