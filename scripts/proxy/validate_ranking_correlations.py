"""
Step 4: Ranking Correlation Validation for Proxy Metrics

Validates proxy scores by computing correlation with ground-truth NAS-Bench-201 accuracy.

Outputs:
  - results/proxy_validation/correlation_results.json  (Spearman rho, Kendall tau, top-K precision)
  - results/proxy_validation/correlation_table.csv
  - results/proxy_validation/scatter_plots.png         (4x2: raw scatter + rank scatter per proxy)
  - results/proxy_validation/pairwise_heatmap.png      (proxy<->proxy redundancy heatmap)
  - results/proxy_validation/accuracy_distribution.png (ground-truth accuracy histogram)
  - results/proxy_validation/topk_precision.png        (top-1%/5%/10% precision bar chart)
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

sns.set_style("whitegrid")
plt.rcParams['font.size'] = 10


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
    """Compute Spearman rho, Kendall tau, and top-K precision."""
    valid_mask = np.isfinite(scores) & np.isfinite(ground_truth)
    valid_scores = scores[valid_mask]
    valid_gt = ground_truth[valid_mask]

    if len(valid_scores) < 2:
        return {'proxy_name': proxy_name, 'count': 0,
                'spearman_rho': np.nan, 'spearman_pvalue': np.nan,
                'kendall_tau': np.nan, 'kendall_pvalue': np.nan,
                'interpretation': 'N/A',
                'topk_precision_1pct': np.nan,
                'topk_precision_5pct': np.nan,
                'topk_precision_10pct': np.nan}

    spearman_rho, spearman_p = spearmanr(valid_scores, valid_gt)
    kendall_tau, kendall_p = kendalltau(valid_scores, valid_gt)

    # Top-K precision: fraction of true top-K that appear in predicted top-K
    n = len(valid_gt)
    topk_precisions = {}
    for pct in [0.01, 0.05, 0.10]:
        k = max(1, int(n * pct))
        true_topk  = set(np.argsort(valid_gt)[-k:])
        pred_topk  = set(np.argsort(valid_scores)[-k:])
        topk_precisions[pct] = len(true_topk & pred_topk) / k

    return {
        'proxy_name': proxy_name,
        'count': len(valid_scores),
        'spearman_rho': float(spearman_rho),
        'spearman_pvalue': float(spearman_p),
        'kendall_tau': float(kendall_tau),
        'kendall_pvalue': float(kendall_p),
        'interpretation': _interpret_correlation(spearman_rho),
        'topk_precision_1pct':  float(topk_precisions[0.01]),
        'topk_precision_5pct':  float(topk_precisions[0.05]),
        'topk_precision_10pct': float(topk_precisions[0.10]),
    }


def _interpret_correlation(rho: float) -> str:
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
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*80)
    print("STEP 4: RANKING CORRELATION VALIDATION")
    print("="*80)

    # ===== LOAD GROUND TRUTH =====
    print("\n[1/4] Loading ground-truth CIFAR-10 accuracy...")
    try:
        ground_truth = load_ground_truth_accuracy(arch_data_dir)
        print(f"  ✓ Loaded {len(ground_truth)} ground-truth accuracies")
    except Exception as e:
        print(f"  ✗ Error loading ground truth: {e}")
        return

    gt_array_full = np.array([ground_truth[k] for k in sorted(ground_truth.keys())])

    # ===== LOAD PROXIES AND COMPUTE CORRELATIONS =====
    print("\n[2/4] Computing correlations with ground truth...")

    proxy_names = ['synflow', 'naswot', 'zenscore', 'param_count']
    all_correlations = {}
    correlation_data = []

    # Storage for scatter / heatmap data
    raw_scores_aligned   = {}   # proxy_name -> np.array aligned to sorted common_ids
    trans_scores_aligned = {}
    gt_aligned           = {}   # proxy_name -> gt np.array (same alignment)

    print("\n  RAW PROXIES:")
    for proxy_name in proxy_names:
        proxy_file = os.path.join(proxy_scores_dir, f'{proxy_name}_full.json')
        if not os.path.exists(proxy_file):
            print(f"    ⚠  {proxy_name}_full.json not found")
            continue
        try:
            proxy_scores = load_proxy_scores(proxy_file)
            common_ids   = sorted(set(ground_truth.keys()) & set(proxy_scores.keys()))
            gt_arr       = np.array([ground_truth[i]   for i in common_ids])
            sc_arr       = np.array([proxy_scores[i]   for i in common_ids])
            raw_scores_aligned[proxy_name] = sc_arr
            gt_aligned[proxy_name]         = gt_arr
            corr = compute_correlations(sc_arr, gt_arr, f'{proxy_name}_raw')
            all_correlations[f'{proxy_name}_raw'] = corr
            correlation_data.append(corr)
            print(f"    {proxy_name:15} | ρ={corr['spearman_rho']:7.4f} | τ={corr['kendall_tau']:7.4f}"
                  f" | top1%={corr['topk_precision_1pct']:.2f}"
                  f" | top5%={corr['topk_precision_5pct']:.2f}"
                  f" | {corr['interpretation']}")
        except Exception as e:
            print(f"    ✗ Error with {proxy_name}: {e}")

    print("\n  TRANSFORMED PROXIES:")
    for proxy_name in proxy_names:
        proxy_file = os.path.join(transformed_dir, f'{proxy_name}_transformed.json')
        if not os.path.exists(proxy_file):
            print(f"    ℹ  {proxy_name}_transformed.json not found")
            continue
        try:
            proxy_scores = load_proxy_scores(proxy_file)
            common_ids   = sorted(set(ground_truth.keys()) & set(proxy_scores.keys()))
            gt_arr       = np.array([ground_truth[i]   for i in common_ids])
            sc_arr       = np.array([proxy_scores[i]   for i in common_ids])
            trans_scores_aligned[proxy_name] = sc_arr
            corr = compute_correlations(sc_arr, gt_arr, f'{proxy_name}_transformed')
            all_correlations[f'{proxy_name}_transformed'] = corr
            correlation_data.append(corr)
            print(f"    {proxy_name:15} | ρ={corr['spearman_rho']:7.4f} | τ={corr['kendall_tau']:7.4f}"
                  f" | top1%={corr['topk_precision_1pct']:.2f}"
                  f" | top5%={corr['topk_precision_5pct']:.2f}"
                  f" | {corr['interpretation']}")
        except Exception as e:
            print(f"    ✗ Error with {proxy_name}: {e}")

    # ===== SAVE JSON & CSV =====
    print("\n[3/4] Saving results...")
    results_file = os.path.join(output_dir, 'correlation_results.json')
    with open(results_file, 'w') as f:
        json.dump(all_correlations, f, indent=2)
    print(f"  ✓ Results saved to: {results_file}")

    df_corr = pd.DataFrame(correlation_data)
    csv_file = os.path.join(output_dir, 'correlation_table.csv')
    df_corr.to_csv(csv_file, index=False)
    print(f"  ✓ Table saved to: {csv_file}")

    # ===== PLOT 1: SCATTER PLOTS (4×2: raw top row, rank-rank bottom row) =====
    print("\n[4/4] Generating visualizations...")

    display_names = {'synflow': 'SynFlow', 'naswot': 'NASWOT',
                     'zenscore': 'Zen-Score', 'param_count': 'Param Count'}
    plot_proxies = [p for p in proxy_names if p in raw_scores_aligned]

    fig, axes = plt.subplots(2, len(plot_proxies), figsize=(5 * len(plot_proxies), 9))
    fig.suptitle('Proxy vs Ground-Truth Accuracy (CIFAR-10)', fontsize=14, fontweight='bold', y=1.01)

    for col, proxy_name in enumerate(plot_proxies):
        sc  = raw_scores_aligned[proxy_name]
        gt  = gt_aligned[proxy_name]
        corr_key = f'{proxy_name}_raw'
        rho = all_correlations[corr_key]['spearman_rho']
        p   = all_correlations[corr_key]['spearman_pvalue']

        # Top row: raw scatter
        ax = axes[0, col]
        ax.scatter(sc, gt, alpha=0.25, s=8, color='steelblue', rasterized=True)
        finite = np.isfinite(sc) & np.isfinite(gt)
        if finite.sum() > 1:
            z = np.polyfit(sc[finite], gt[finite], 1)
            x_line = np.linspace(sc[finite].min(), sc[finite].max(), 200)
            ax.plot(x_line, np.poly1d(z)(x_line), 'r--', linewidth=1.5, alpha=0.9)
        ax.set_title(f'{display_names[proxy_name]} — Raw\nρ = {rho:.4f}  (p={p:.2e})',
                     fontweight='bold', fontsize=10)
        ax.set_xlabel('Proxy Score')
        ax.set_ylabel('CIFAR-10 Accuracy (%)')
        ax.grid(True, alpha=0.3)

        # Bottom row: rank-rank scatter (transformed if available, else raw)
        ax2 = axes[1, col]
        if proxy_name in trans_scores_aligned:
            sc2 = trans_scores_aligned[proxy_name]
            common_ids2 = sorted(set(ground_truth.keys()) &
                                 set({str(k): None for k in range(len(sc2))}.keys())
                                 if False else set(ground_truth.keys()))
            # Realign transformed to same gt
            t_file = os.path.join(transformed_dir, f'{proxy_name}_transformed.json')
            with open(t_file) as f_:
                t_data = json.load(f_)
            t_raw = t_data['results'] if 'results' in t_data else t_data
            common = sorted(set(ground_truth.keys()) & {int(k) for k in t_raw.keys()})
            sc2 = np.array([float(t_raw[str(i)]) for i in common])
            gt2 = np.array([ground_truth[i] for i in common])
            corr_key2 = f'{proxy_name}_transformed'
            rho2 = all_correlations.get(corr_key2, {}).get('spearman_rho', float('nan'))
            label = 'Transformed'
            color2 = 'darkorange'
        else:
            sc2, gt2, rho2, label, color2 = sc, gt, rho, 'Raw', 'steelblue'

        finite2 = np.isfinite(sc2) & np.isfinite(gt2)
        sc2_rank = np.argsort(np.argsort(sc2[finite2]))
        gt2_rank = np.argsort(np.argsort(gt2[finite2]))
        ax2.scatter(sc2_rank, gt2_rank, alpha=0.2, s=8, color=color2, rasterized=True)
        z2 = np.polyfit(sc2_rank, gt2_rank, 1)
        ax2.plot(np.linspace(sc2_rank.min(), sc2_rank.max(), 200),
                 np.poly1d(z2)(np.linspace(sc2_rank.min(), sc2_rank.max(), 200)),
                 'r--', linewidth=1.5, alpha=0.9)
        ax2.set_title(f'{display_names[proxy_name]} — Rank ({label})\nρ = {rho2:.4f}',
                      fontweight='bold', fontsize=10)
        ax2.set_xlabel('Proxy Rank')
        ax2.set_ylabel('Accuracy Rank')
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    scatter_file = os.path.join(output_dir, 'scatter_plots.png')
    plt.savefig(scatter_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Scatter plots saved to: {scatter_file}")

    # ===== PLOT 2: PAIRWISE INTER-PROXY HEATMAP =====
    # Use transformed scores where available, else raw
    heatmap_data = {}
    for p in proxy_names:
        if p in trans_scores_aligned:
            heatmap_data[display_names[p]] = trans_scores_aligned[p]
        elif p in raw_scores_aligned:
            heatmap_data[display_names[p]] = raw_scores_aligned[p]

    # All arrays may differ in length due to missing archs — align to intersection
    all_ids_sets = []
    for p in proxy_names:
        src_dir  = transformed_dir if p in trans_scores_aligned else proxy_scores_dir
        suffix   = '_transformed' if p in trans_scores_aligned else '_full'
        fname    = os.path.join(src_dir, f'{p}{suffix}.json')
        if os.path.exists(fname):
            with open(fname) as f_:
                d = json.load(f_)
            r = d['results'] if 'results' in d else d
            all_ids_sets.append({int(k) for k in r.keys()})
    common_heatmap_ids = sorted(set.intersection(*all_ids_sets)) if all_ids_sets else []

    if len(common_heatmap_ids) > 10:
        hm_vectors = {}
        for p in proxy_names:
            src_dir = transformed_dir if p in trans_scores_aligned else proxy_scores_dir
            suffix  = '_transformed' if p in trans_scores_aligned else '_full'
            fname   = os.path.join(src_dir, f'{p}{suffix}.json')
            if os.path.exists(fname):
                with open(fname) as f_:
                    d = json.load(f_)
                r = d['results'] if 'results' in d else d
                hm_vectors[display_names[p]] = np.array([float(r[str(i)]) for i in common_heatmap_ids])

        # Add ground truth
        hm_vectors['GT Accuracy'] = np.array([ground_truth[i] for i in common_heatmap_ids
                                               if i in ground_truth])
        # Trim to shortest (GT may be shorter)
        min_len = min(len(v) for v in hm_vectors.values())
        hm_vectors = {k: v[:min_len] for k, v in hm_vectors.items()}

        labels = list(hm_vectors.keys())
        n_vars = len(labels)
        rho_matrix = np.ones((n_vars, n_vars))
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    r, _ = spearmanr(hm_vectors[labels[i]], hm_vectors[labels[j]])
                    rho_matrix[i, j] = r

        fig, ax = plt.subplots(figsize=(7, 6))
        mask = np.triu(np.ones_like(rho_matrix, dtype=bool), k=1)
        sns.heatmap(rho_matrix, annot=True, fmt='.2f', cmap='RdYlGn',
                    vmin=-1, vmax=1, center=0,
                    xticklabels=labels, yticklabels=labels,
                    linewidths=0.5, ax=ax, square=True,
                    annot_kws={'size': 10})
        ax.set_title('Pairwise Spearman ρ — Proxies + Ground Truth\n(transformed scores)',
                     fontweight='bold', fontsize=11)
        plt.tight_layout()
        heatmap_file = os.path.join(output_dir, 'pairwise_heatmap.png')
        plt.savefig(heatmap_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Pairwise heatmap saved to: {heatmap_file}")

    # ===== PLOT 3: ACCURACY DISTRIBUTION =====
    fig, axes_acc = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Ground-Truth CIFAR-10 Accuracy Distribution (NAS-Bench-201, epoch 199)',
                 fontweight='bold', fontsize=12)

    # Left: full histogram
    ax = axes_acc[0]
    ax.hist(gt_array_full, bins=100, color='steelblue', alpha=0.8, edgecolor='none')
    ax.axvline(np.percentile(gt_array_full, 10), color='red',   linestyle='--', linewidth=1.2, label='10th pct')
    ax.axvline(np.median(gt_array_full),          color='orange', linestyle='--', linewidth=1.2, label='Median')
    ax.axvline(np.percentile(gt_array_full, 90),  color='green', linestyle='--', linewidth=1.2, label='90th pct')
    ax.set_xlabel('Test Accuracy (%)')
    ax.set_ylabel('Count')
    ax.set_title('Full distribution')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: zoom into top 10% to show elite architecture spread
    threshold_90 = np.percentile(gt_array_full, 90)
    elite = gt_array_full[gt_array_full >= threshold_90]
    ax2 = axes_acc[1]
    ax2.hist(elite, bins=50, color='seagreen', alpha=0.8, edgecolor='none')
    ax2.set_xlabel('Test Accuracy (%)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Top 10% architectures (n={len(elite):,}, acc ≥ {threshold_90:.1f}%)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    acc_dist_file = os.path.join(output_dir, 'accuracy_distribution.png')
    plt.savefig(acc_dist_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Accuracy distribution saved to: {acc_dist_file}")

    # ===== PLOT 4: TOP-K PRECISION BAR CHART =====
    topk_proxies = [p for p in proxy_names if f'{p}_transformed' in all_correlations
                    or f'{p}_raw' in all_correlations]
    topk_labels, p1, p5, p10 = [], [], [], []
    for p in topk_proxies:
        key = f'{p}_transformed' if f'{p}_transformed' in all_correlations else f'{p}_raw'
        c = all_correlations[key]
        topk_labels.append(display_names[p])
        p1.append(c['topk_precision_1pct'])
        p5.append(c['topk_precision_5pct'])
        p10.append(c['topk_precision_10pct'])

    x = np.arange(len(topk_labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width,   p1,  width, label='Top 1%',  color='steelblue',  alpha=0.85)
    ax.bar(x,           p5,  width, label='Top 5%',  color='darkorange', alpha=0.85)
    ax.bar(x + width,   p10, width, label='Top 10%', color='seagreen',   alpha=0.85)
    ax.axhline(1.0, color='black', linestyle='--', linewidth=0.8, alpha=0.5, label='Perfect precision')
    ax.set_xticks(x)
    ax.set_xticklabels(topk_labels, fontsize=11)
    ax.set_ylabel('Precision (fraction of true top-K recovered)')
    ax.set_ylim(0, 1.15)
    ax.set_title('Top-K Precision: Fraction of Elite Architectures Correctly Identified',
                 fontweight='bold', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    for bar_group in [x - width, x, x + width]:
        for xi, val in zip(bar_group, [p1, p5, p10][[list(x - width), list(x), list(x + width)].index(list(bar_group))]):
            ax.text(xi, val + 0.02, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    topk_file = os.path.join(output_dir, 'topk_precision.png')
    plt.savefig(topk_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Top-K precision chart saved to: {topk_file}")

    # ===== PRINT SUMMARY =====
    print("\n" + "="*80)
    print("CORRELATION SUMMARY")
    print("="*80)
    print("\nKey Thresholds:  ρ ≥ 0.6 Strong | 0.3–0.6 Moderate | <0.3 Weak\n")

    for key, corr in all_correlations.items():
        print(f"{key.upper()}")
        print(f"  Spearman ρ: {corr['spearman_rho']:8.4f}  (p={corr['spearman_pvalue']:.2e})  "
              f"Kendall τ: {corr['kendall_tau']:7.4f}  [{corr['interpretation']}]")
        print(f"  Top-K precision — 1%: {corr['topk_precision_1pct']:.3f}  "
              f"5%: {corr['topk_precision_5pct']:.3f}  "
              f"10%: {corr['topk_precision_10pct']:.3f}")

    # ===== DECISION POINT =====
    print("\n" + "="*80)
    print("CRITICAL DECISION POINT")
    print("="*80)

    t_synflow  = all_correlations.get('synflow_transformed',     {}).get('spearman_rho', 0)
    t_naswot   = all_correlations.get('naswot_transformed',      {}).get('spearman_rho', 0)
    t_zen      = all_correlations.get('zenscore_transformed',    {}).get('spearman_rho', 0)
    t_params   = all_correlations.get('param_count_transformed', {}).get('spearman_rho', 0)

    print(f"\nTransformed proxy ranking signals:")
    for name, rho in [('SynFlow', t_synflow), ('NASWOT', t_naswot),
                      ('Zen-Score', t_zen), ('Param Count', t_params)]:
        print(f"  {name:12} ρ = {rho:.4f}  [{_interpret_correlation(rho)}]")

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
