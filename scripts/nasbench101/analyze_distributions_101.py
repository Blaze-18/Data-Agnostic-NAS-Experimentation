"""
Step 3: Analyze proxy distributions for NAS-Bench-101.

Computes per-proxy distribution statistics and saves summary JSON + plots.

Input:  results/nasbench101/transformed_proxy/
Output: results/nasbench101/proxy_distribution/
        - stats_summary.json
        - distribution_plots.png (4-panel histogram grid)
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import skew, kurtosis

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, skipping plots")

ROOT_DIR  = Path(__file__).resolve().parents[2]
TRANS_DIR = ROOT_DIR / "results/nasbench101/transformed_proxy"
OUT_DIR   = ROOT_DIR / "results/nasbench101/proxy_distribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PROXIES = [
    ("param_count", "param_count_log.npy"),
    ("synflow",     "synflow_log.npy"),
    ("naswot",      "naswot_log.npy"),
    ("zenscore",    "zenscore_log.npy"),
]


def describe(arr: np.ndarray, name: str) -> dict:
    arr = arr.astype(np.float64)
    finite = arr[np.isfinite(arr)]
    n_finite = len(finite)
    n_inf    = int(np.sum(~np.isfinite(arr)))
    result = {
        "n_total":   int(len(arr)),
        "n_finite":  n_finite,
        "n_nonfinite": n_inf,
        "mean":      float(np.mean(finite)),
        "std":       float(np.std(finite)),
        "min":       float(np.min(finite)),
        "p5":        float(np.percentile(finite, 5)),
        "p25":       float(np.percentile(finite, 25)),
        "median":    float(np.median(finite)),
        "p75":       float(np.percentile(finite, 75)),
        "p95":       float(np.percentile(finite, 95)),
        "max":       float(np.max(finite)),
        "skewness":  float(skew(finite)),
        "kurtosis":  float(kurtosis(finite)),
        "normalized": bool(abs(skew(finite)) <= 2.0),
    }
    flag = "" if result["normalized"] else "  *** NOT NORMALIZED (|skewness| > 2) ***"
    print(f"  {name:15s}: mean={result['mean']:.3f}  std={result['std']:.3f}  "
          f"skew={result['skewness']:.3f}  kurt={result['kurtosis']:.3f}{flag}", flush=True)
    return result


def main():
    summary = {}
    proxy_data = {}
    
    print("Distribution analysis:\n", flush=True)
    for name, fname in PROXIES:
        path = TRANS_DIR / fname
        if not path.exists():
            print(f"  {name}: FILE NOT FOUND -- skipping", flush=True)
            continue
        arr = np.load(path)
        summary[name] = describe(arr, name)
        proxy_data[name] = arr

    with open(OUT_DIR / "stats_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved stats_summary.json to {OUT_DIR}", flush=True)
    
    # Generate plots if matplotlib is available
    if HAS_MATPLOTLIB and proxy_data:
        print("\nGenerating distribution plots...", flush=True)
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('NAS-Bench-101 Proxy Distributions (Log-Transformed)', fontsize=16, fontweight='bold')
        
        proxy_names = ['param_count', 'synflow', 'naswot', 'zenscore']
        titles = ['Param Count (log)', 'SynFlow (log)', 'NASWOT (-log)', 'ZenScore (-log)']
        
        for idx, (name, title) in enumerate(zip(proxy_names, titles)):
            if name not in proxy_data:
                continue
            
            ax = axes[idx // 2, idx % 2]
            data = proxy_data[name]
            
            # Histogram
            ax.hist(data, bins=100, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
            
            # Add statistics
            stats = summary[name]
            textstr = f"Mean: {stats['mean']:.2f}\nStd: {stats['std']:.2f}\n"
            textstr += f"Skew: {stats['skewness']:.2f}\nKurt: {stats['kurtosis']:.2f}"
            
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.72, 0.97, textstr, transform=ax.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
            
            ax.set_xlabel('Value', fontsize=11)
            ax.set_ylabel('Frequency', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plot_path = OUT_DIR / "distribution_plots.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved distribution_plots.png to {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
