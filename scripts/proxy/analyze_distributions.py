"""
Step 3: Distribution Analysis for Proxy Metrics

Analyzes raw and (optionally) log-transformed proxy score distributions.
Computes statistical summaries and generates comprehensive visualization plots.

Output:
  - results/proxy_distribution/stats_summary.json
  - results/proxy_distribution/distribution_plots.png (histograms, KDE, box plots)
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)


def load_proxy_scores(proxy_file: str) -> np.ndarray:
    """Load proxy scores from JSON file."""
    with open(proxy_file, 'r') as f:
        data = json.load(f)
    
    # Extract scores from 'results' dict
    if isinstance(data, dict) and 'results' in data:
        scores_dict = data['results']
    else:
        scores_dict = data
    
    # Convert to numpy array (values only, preserving order)
    scores = np.array([float(v) for v in scores_dict.values()])
    return scores


def compute_statistics(scores: np.ndarray, proxy_name: str) -> Dict:
    """
    Compute comprehensive statistics for a proxy metric.
    
    Returns:
        Dictionary with statistical measures
    """
    # Filter out NaNs and infs
    valid_scores = scores[np.isfinite(scores)]
    
    # Count near-zero values (defined as < 1e-6 after log transform for comparison)
    near_zero_count = np.sum(valid_scores < 1e-6)
    near_zero_pct = (near_zero_count / len(valid_scores)) * 100 if len(valid_scores) > 0 else 0
    
    stats = {
        'proxy_name': proxy_name,
        'count': len(valid_scores),
        'mean': float(np.mean(valid_scores)),
        'median': float(np.median(valid_scores)),
        'std': float(np.std(valid_scores)),
        'min': float(np.min(valid_scores)),
        'max': float(np.max(valid_scores)),
        'q25': float(np.percentile(valid_scores, 25)),
        'q75': float(np.percentile(valid_scores, 75)),
        'skewness': float(np.mean((valid_scores - np.mean(valid_scores))**3) / (np.std(valid_scores)**3 + 1e-10)),
        'kurtosis': float(np.mean((valid_scores - np.mean(valid_scores))**4) / (np.std(valid_scores)**4 + 1e-10)),
        'near_zero_count': int(near_zero_count),
        'near_zero_percentage': float(near_zero_pct),
        'range': float(np.max(valid_scores) - np.min(valid_scores)),
        'cv': float(np.std(valid_scores) / (np.mean(valid_scores) + 1e-10))  # Coefficient of variation
    }
    
    return stats


def analyze_distributions(output_dir: str = 'results/proxy_distribution'):
    """
    Main analysis function: Load proxies, compute stats, generate plots.
    
    Args:
        output_dir: Directory to save results
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    proxy_scores_dir = 'results/raw_proxy_scores'
    transformed_dir = 'results/transformed_proxy'
    
    # Proxy files to analyze
    proxy_names = ['synflow', 'naswot', 'zenscore', 'param_count']
    
    # Storage for all data
    all_stats = {}
    all_data = {}  # Store for plotting
    
    print("\n" + "="*80)
    print("STEP 3: DISTRIBUTION ANALYSIS FOR PROXY METRICS")
    print("="*80)
    
    # ===== LOAD RAW PROXIES =====
    print("\n[1/3] Loading RAW proxy scores...")
    raw_data = {}
    
    for proxy_name in proxy_names:
        proxy_file = os.path.join(proxy_scores_dir, f'{proxy_name}_full.json')
        
        if not os.path.exists(proxy_file):
            print(f"  ⚠️  {proxy_name}_full.json not found, skipping")
            continue
        
        try:
            scores = load_proxy_scores(proxy_file)
            raw_data[proxy_name] = scores
            stats = compute_statistics(scores, f'{proxy_name}_raw')
            all_stats[f'{proxy_name}_raw'] = stats
            
            print(f"  ✓ {proxy_name:15} | n={stats['count']}, mean={stats['mean']:.6e}, "
                  f"std={stats['std']:.6e}, near_zero={stats['near_zero_percentage']:.1f}%")
        
        except Exception as e:
            print(f"  ✗ Error loading {proxy_name}: {e}")
    
    # ===== LOAD TRANSFORMED PROXIES =====
    print("\n[2/3] Loading TRANSFORMED proxy scores...")
    transformed_data = {}
    
    for proxy_name in proxy_names:
        proxy_file = os.path.join(transformed_dir, f'{proxy_name}_transformed.json')
        
        if not os.path.exists(proxy_file):
            print(f"  ℹ  {proxy_name}_transformed.json not found (Step 2 not run yet)")
            continue
        
        try:
            scores = load_proxy_scores(proxy_file)
            transformed_data[proxy_name] = scores
            stats = compute_statistics(scores, f'{proxy_name}_transformed')
            all_stats[f'{proxy_name}_transformed'] = stats
            
            print(f"  ✓ {proxy_name:15} | n={stats['count']}, mean={stats['mean']:.6e}, "
                  f"std={stats['std']:.6e}, near_zero={stats['near_zero_percentage']:.1f}%")
        
        except Exception as e:
            print(f"  ✗ Error loading {proxy_name}: {e}")
    
    # ===== SAVE STATISTICS SUMMARY =====
    print("\n[3/3] Generating visualizations and saving statistics...")
    
    stats_file = os.path.join(output_dir, 'stats_summary.json')
    with open(stats_file, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"  ✓ Statistics saved to: {stats_file}")
    
    # ===== CREATE VISUALIZATIONS — split into two files =====

    col_idx = {'synflow': 0, 'naswot': 1, 'zenscore': 2, 'param_count': 3}

    # --- IMAGE 1: RAW distributions (histogram + KDE + box plot) ---
    if raw_data:
        fig, axes = plt.subplots(3, 4, figsize=(18, 10))
        fig.suptitle('Proxy Score Distributions — RAW', fontsize=15, fontweight='bold')

        for proxy_name, scores in raw_data.items():
            ci = col_idx[proxy_name]
            valid = scores[np.isfinite(scores)]

            # Row 0: histogram (log-scale y)
            ax = axes[0, ci]
            ax.hist(valid, bins=100, color='steelblue', alpha=0.75, edgecolor='none')
            ax.set_title(f'{proxy_name.upper()}', fontweight='bold')
            ax.set_ylabel('Frequency')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)

            # Row 1: KDE
            ax = axes[1, ci]
            ax.hist(valid, bins=100, density=True, alpha=0.45, color='steelblue', label='Histogram')
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(valid)
                x_range = np.linspace(valid.min(), valid.max(), 300)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except Exception:
                pass
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # Row 2: box plot
            ax = axes[2, ci]
            ax.boxplot(valid, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='steelblue', alpha=0.5))
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        raw_plot_file = os.path.join(output_dir, 'distribution_plots_raw.png')
        plt.savefig(raw_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Raw distribution plots saved to: {raw_plot_file}")

    # --- IMAGE 2: TRANSFORMED distributions (histogram + KDE + box plot) ---
    if transformed_data:
        fig, axes = plt.subplots(3, 4, figsize=(18, 10))
        fig.suptitle('Proxy Score Distributions — TRANSFORMED (log)', fontsize=15, fontweight='bold')

        for proxy_name, scores in transformed_data.items():
            ci = col_idx[proxy_name]
            valid = scores[np.isfinite(scores)]

            ax = axes[0, ci]
            ax.hist(valid, bins=100, color='darkgreen', alpha=0.75, edgecolor='none')
            ax.set_title(f'{proxy_name.upper()}', fontweight='bold')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)

            ax = axes[1, ci]
            ax.hist(valid, bins=100, density=True, alpha=0.45, color='darkgreen', label='Histogram')
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(valid)
                x_range = np.linspace(valid.min(), valid.max(), 300)
                ax.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
            except Exception:
                pass
            ax.set_ylabel('Density')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            ax = axes[2, ci]
            ax.boxplot(valid, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='darkgreen', alpha=0.5))
            ax.set_ylabel('Score')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        trans_plot_file = os.path.join(output_dir, 'distribution_plots_transformed.png')
        plt.savefig(trans_plot_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Transformed distribution plots saved to: {trans_plot_file}")
    
    # ===== PRINT SUMMARY TABLE =====
    print("\n" + "="*80)
    print("DISTRIBUTION STATISTICS SUMMARY")
    print("="*80)
    
    for key, stats in all_stats.items():
        print(f"\n{key.upper()}")
        print(f"  Count:           {stats['count']}")
        print(f"  Mean:            {stats['mean']:.6e}")
        print(f"  Median:          {stats['median']:.6e}")
        print(f"  Std:             {stats['std']:.6e}")
        print(f"  Min - Max:       {stats['min']:.6e} - {stats['max']:.6e}")
        print(f"  Range:           {stats['range']:.6e}")
        print(f"  Skewness:        {stats['skewness']:.4f}")
        print(f"  Kurtosis:        {stats['kurtosis']:.4f}")
        print(f"  Coeff. Variation: {stats['cv']:.4f}")
        print(f"  Near-zero (< 1e-6): {stats['near_zero_percentage']:.2f}% ({stats['near_zero_count']} values)")
    
    print("\n" + "="*80)
    print("✓ Distribution analysis complete!")
    print("="*80 + "\n")
    
    return all_stats


if __name__ == '__main__':
    analyze_distributions()
