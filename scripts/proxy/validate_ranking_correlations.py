"""
Step 4: Ranking Correlation Validation for Proxy Metrics

Validates proxy scores by computing correlation with ground-truth NAS-Bench-201 accuracy.
Computes Spearman ρ and Kendall τ rank correlations.

Output:
  - results/proxy_validation/correlation_results.json
  - results/proxy_validation/correlation_plots.png
  - results/proxy_validation/correlation_table.csv
"""

import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple
from scipy.stats import spearmanr, kendalltau

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)


def load_ground_truth_accuracy(arch_data_dir: str = 'data/chunks_clean/arch2infos', 
                               dataset: str = 'cifar10', 
                               eval_type: str = 'ori-test',
                               epoch: int = 199) -> Dict[int, float]:
    """
    Load ground-truth accuracy from NAS-Bench-201 architecture files.
    
    Args:
        arch_data_dir: Directory with architecture files
        dataset: Dataset name ('cifar10', 'cifar100', 'imagenet16-120')
        eval_type: Evaluation type ('ori-test', 'x-valid')
        epoch: Epoch number (default: 199, final epoch)
    
    Returns:
        Dictionary mapping arch_id -> test_accuracy
    """
    
    accuracy_dict = {}
    arch_files = sorted([f for f in os.listdir(arch_data_dir) if f.endswith('.pth')])
    
    for arch_file in arch_files:
        arch_id = int(arch_file.replace('.pth', ''))
        arch_path = os.path.join(arch_data_dir, arch_file)
        
        try:
            data = torch.load(arch_path, weights_only=False)
            arch_data = list(data.values())[0]  # Get first (only) architecture
            
            # Extract accuracy from all_results
            all_results = arch_data['full']['all_results']
            
            # Look for the dataset and seed combination
            acc_value = None
            for (dset_name, seed), results in all_results.items():
                if dset_name.lower() == dataset.lower():
                    # Get final epoch accuracy
                    eval_key = f'{eval_type}@{epoch}'
                    if 'eval_acc1es' in results and eval_key in results['eval_acc1es']:
                        acc_value = results['eval_acc1es'][eval_key]
                        break
            
            if acc_value is not None:
                accuracy_dict[arch_id] = float(acc_value)
        
        except Exception as e:
            continue
    
    return accuracy_dict


def load_proxy_scores(proxy_file: str) -> Dict[int, float]:
    """Load proxy scores from JSON file."""
    with open(proxy_file, 'r') as f:
        data = json.load(f)
    
    if isinstance(data, dict) and 'results' in data:
        scores_dict = data['results']
    else:
        scores_dict = data
    
    # Convert string keys to int
    return {int(k): float(v) for k, v in scores_dict.items()}


def compute_correlations(scores: np.ndarray, ground_truth: np.ndarray, 
                        proxy_name: str) -> Dict:
    """
    Compute Spearman and Kendall correlations between proxy and ground truth.
    
    Args:
        scores: Proxy scores
        ground_truth: Ground-truth accuracy values
        proxy_name: Name of proxy
    
    Returns:
        Dictionary with correlation results
    """
    
    # Remove NaNs/infs
    valid_mask = np.isfinite(scores) & np.isfinite(ground_truth)
    valid_scores = scores[valid_mask]
    valid_gt = ground_truth[valid_mask]
    
    if len(valid_scores) < 2:
        return {
            'proxy_name': proxy_name,
            'count': 0,
            'spearman_rho': np.nan,
            'spearman_pvalue': np.nan,
            'kendall_tau': np.nan,
            'kendall_pvalue': np.nan
        }
    
    # Compute Spearman rank correlation
    spearman_rho, spearman_p = spearmanr(valid_scores, valid_gt)
    
    # Compute Kendall Tau correlation
    kendall_tau, kendall_p = kendalltau(valid_scores, valid_gt)
    
    return {
        'proxy_name': proxy_name,
        'count': len(valid_scores),
        'spearman_rho': float(spearman_rho),
        'spearman_pvalue': float(spearman_p),
        'kendall_tau': float(kendall_tau),
        'kendall_pvalue': float(kendall_p),
        'interpretation': _interpret_correlation(spearman_rho)
    }


def _interpret_correlation(rho: float) -> str:
    """Interpret Spearman correlation strength."""
    abs_rho = abs(rho)
    if abs_rho >= 0.6:
        return "Strong"
    elif abs_rho >= 0.3:
        return "Moderate"
    elif abs_rho > 0.0:
        return "Weak"
    else:
        return "No signal"


def validate_rankings(output_dir: str = 'results/proxy_validation',
                    arch_data_dir: str = 'data/chunks_clean/arch2infos',
                    proxy_scores_dir: str = 'results/raw_proxy_scores',
                    transformed_dir: str = 'results/transformed_proxy'):
    """
    Main validation function: Compute correlations with ground truth.
    
    Args:
        output_dir: Directory to save results
        arch_data_dir: Directory with architecture files
        proxy_scores_dir: Directory with raw proxy scores
        transformed_dir: Directory with transformed proxy scores
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("STEP 4: RANKING CORRELATION VALIDATION")
    print("="*80)
    
    # ===== LOAD GROUND TRUTH =====
    print("\n[1/3] Loading ground-truth CIFAR-10 accuracy...")
    try:
        ground_truth = load_ground_truth_accuracy(arch_data_dir)
        print(f"  ✓ Loaded {len(ground_truth)} ground-truth accuracies")
    except Exception as e:
        print(f"  ✗ Error loading ground truth: {e}")
        return
    
    # ===== LOAD PROXIES AND COMPUTE CORRELATIONS =====
    print("\n[2/3] Computing correlations with ground truth...")
    
    proxy_names = ['synflow', 'naswot', 'zenscore', 'param_count']
    all_correlations = {}
    correlation_data = []  # For DataFrame
    
    # Process raw proxies
    print("\n  RAW PROXIES:")
    for proxy_name in proxy_names:
        proxy_file = os.path.join(proxy_scores_dir, f'{proxy_name}_full.json')
        
        if not os.path.exists(proxy_file):
            print(f"    ⚠️  {proxy_name}_full.json not found")
            continue
        
        try:
            proxy_scores = load_proxy_scores(proxy_file)
            
            # Align with ground truth (only use architectures in both)
            common_ids = set(ground_truth.keys()) & set(proxy_scores.keys())
            gt_array = np.array([ground_truth[id] for id in sorted(common_ids)])
            scores_array = np.array([proxy_scores[id] for id in sorted(common_ids)])
            
            # Compute correlations
            corr_result = compute_correlations(scores_array, gt_array, f'{proxy_name}_raw')
            all_correlations[f'{proxy_name}_raw'] = corr_result
            correlation_data.append(corr_result)
            
            print(f"    {proxy_name:15} | ρ={corr_result['spearman_rho']:7.4f} "
                f"| τ={corr_result['kendall_tau']:7.4f} | {corr_result['interpretation']}")
        
        except Exception as e:
            print(f"    ✗ Error with {proxy_name}: {e}")
    
    # Process transformed proxies (if available)
    print("\n  TRANSFORMED PROXIES:")
    for proxy_name in proxy_names:
        proxy_file = os.path.join(transformed_dir, f'{proxy_name}_transformed.json')
        
        if not os.path.exists(proxy_file):
            print(f"    ℹ  {proxy_name}_transformed.json not found (Step 2 not run)")
            continue
        
        try:
            proxy_scores = load_proxy_scores(proxy_file)
            
            # Align with ground truth
            common_ids = set(ground_truth.keys()) & set(proxy_scores.keys())
            gt_array = np.array([ground_truth[id] for id in sorted(common_ids)])
            scores_array = np.array([proxy_scores[id] for id in sorted(common_ids)])
            
            # Compute correlations
            corr_result = compute_correlations(scores_array, gt_array, f'{proxy_name}_transformed')
            all_correlations[f'{proxy_name}_transformed'] = corr_result
            correlation_data.append(corr_result)
            
            print(f"    {proxy_name:15} | ρ={corr_result['spearman_rho']:7.4f} "
                f"| τ={corr_result['kendall_tau']:7.4f} | {corr_result['interpretation']}")
        
        except Exception as e:
            print(f"    ✗ Error with {proxy_name}: {e}")
    
    # ===== SAVE RESULTS =====
    print("\n[3/3] Saving results and generating plots...")
    
    # Save JSON results
    results_file = os.path.join(output_dir, 'correlation_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_correlations, f, indent=2)
    print(f"  ✓ Results saved to: {results_file}")
    
    # Save as CSV for easy viewing
    df_corr = pd.DataFrame(correlation_data)
    csv_file = os.path.join(output_dir, 'correlation_table.csv')
    df_corr.to_csv(csv_file, index=False)
    print(f"  ✓ Table saved to: {csv_file}")
    
    # ===== CREATE VISUALIZATIONS =====
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Proxy Score Ranking Correlation with Ground Truth (CIFAR-10)', 
                fontsize=14, fontweight='bold')
    
    plot_idx = 0
    raw_proxies = [k for k in all_correlations.keys() if k.endswith('_raw')]
    
    for proxy_key in sorted(raw_proxies):
        if plot_idx >= 4:
            break
        
        ax = axes[plot_idx // 2, plot_idx % 2]
        proxy_name = proxy_key.replace('_raw', '')
        
        # Load proxy scores and ground truth
        proxy_file = os.path.join(proxy_scores_dir, f'{proxy_name}_full.json')
        proxy_scores = load_proxy_scores(proxy_file)
        
        # Align
        common_ids = sorted(set(ground_truth.keys()) & set(proxy_scores.keys()))
        scores = np.array([proxy_scores[id] for id in common_ids])
        accuracies = np.array([ground_truth[id] for id in common_ids])
        
        # Scatter plot
        ax.scatter(scores, accuracies, alpha=0.5, s=20)
        
        # Add trend line
        if len(scores) > 1:
            z = np.polyfit(scores, accuracies, 1)
            p = np.poly1d(z)
            x_line = np.linspace(scores.min(), scores.max(), 100)
            ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        # Labels and title
        corr = all_correlations[proxy_key]
        title = (f"{proxy_name.upper()} - Raw\n"
                f"ρ = {corr['spearman_rho']:.4f} (p={corr['spearman_pvalue']:.2e})")
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel('Proxy Score')
        ax.set_ylabel('CIFAR-10 Accuracy')
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, 'correlation_plots.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"  ✓ Plots saved to: {plot_file}")
    
    plt.close()
    
    # ===== PRINT SUMMARY =====
    print("\n" + "="*80)
    print("CORRELATION SUMMARY")
    print("="*80)
    
    print("\nKey Thresholds:")
    print("  ρ ≥ 0.6  → Strong ranking signal")
    print("  0.3 ≤ ρ < 0.6 → Moderate ranking signal")
    print("  0 < ρ < 0.3 → Weak ranking signal")
    print("  ρ ≈ 0 → No ranking signal")
    
    print("\nDetailed Results:")
    for key, corr in all_correlations.items():
        print(f"\n{key.upper()}")
        print(f"  Spearman ρ: {corr['spearman_rho']:8.4f} (p={corr['spearman_pvalue']:.2e})")
        print(f"  Kendall τ:  {corr['kendall_tau']:8.4f} (p={corr['kendall_pvalue']:.2e})")
        print(f"  Interpretation: {corr['interpretation']}")
        print(f"  Count: {corr['count']} architectures")
    
    # ===== CRITICAL DECISION =====
    print("\n" + "="*80)
    print("CRITICAL DECISION POINT")
    print("="*80)
    
    # Evaluate on TRANSFORMED proxies (negation-corrected)
    t_naswot   = all_correlations.get('naswot_transformed',      {}).get('spearman_rho', 0)
    t_zen      = all_correlations.get('zenscore_transformed',    {}).get('spearman_rho', 0)
    t_synflow  = all_correlations.get('synflow_transformed',     {}).get('spearman_rho', 0)
    t_params   = all_correlations.get('param_count_transformed', {}).get('spearman_rho', 0)

    print(f"\nTransformed proxy ranking signals (used for downstream steps):")
    print(f"  SynFlow ρ:     {t_synflow:.4f} - {_interpret_correlation(t_synflow)}")
    print(f"  NASWOT ρ:      {t_naswot:.4f} - {_interpret_correlation(t_naswot)}")
    print(f"  Zen-Score ρ:   {t_zen:.4f} - {_interpret_correlation(t_zen)}")
    print(f"  Param Count ρ: {t_params:.4f} - {_interpret_correlation(t_params)}")

    strong   = [n for n, r in [('SynFlow', t_synflow), ('NASWOT', t_naswot),
                                ('Zen-Score', t_zen), ('Param Count', t_params)] if r >= 0.6]
    moderate = [n for n, r in [('SynFlow', t_synflow), ('NASWOT', t_naswot),
                                ('Zen-Score', t_zen), ('Param Count', t_params)] if 0.3 <= r < 0.6]
    weak     = [n for n, r in [('SynFlow', t_synflow), ('NASWOT', t_naswot),
                                ('Zen-Score', t_zen), ('Param Count', t_params)] if 0 < r < 0.3]

    print()
    if strong:
        print(f"  Strong signal:   {', '.join(strong)}")
    if moderate:
        print(f"  Moderate signal: {', '.join(moderate)}")
    if weak:
        print(f"  Weak signal:     {', '.join(weak)}")

    if len(strong) + len(moderate) >= 3:
        print("\n✓ Sufficient ranking signal — proceed to Step 5 (bias disentanglement)")
    elif len(strong) + len(moderate) >= 1:
        print("\n⚠ Partial ranking signal — address weak proxies, then proceed to Step 5")
    else:
        print("\n✗ No usable ranking signal — metric redesign required before proceeding")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    validate_rankings()
