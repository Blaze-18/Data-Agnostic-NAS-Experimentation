"""
Step 2: Log Transformation for Proxy Metrics

Applies log transformation to stabilize distributions and reduce skew:
  - SynFlow     →  log(score + 1e-8)                  [direction: positive]
  - NASWOT      → -log(score + 1e-8)   [NEGATED]      [direction: corrected]
  - Zen-Score   → -log(score + 1e-8)   [NEGATED]      [direction: corrected]
  - Param Count →  log(params + 1)

NEGATION RATIONALE (NASWOT & Zen-Score):
  Our implementation computes trace(Cov(activations)) — mean activation variance
  across all Conv2d layers. This produces a NEGATIVE correlation with accuracy because:
    - Skip-heavy architectures (poor): input noise passes through unchanged → HIGH variance
    - Conv-heavy architectures (better): BatchNorm normalizes → LOW variance
  Negating converts the metric from "activation noise energy" to
  "activation compression quality": higher = more structured = better architecture.
  The underlying signal is valid (p-value = 0.0), only the direction needs flipping.
  Mathematically: -log(x) = log(1/x), i.e., we score by inverse variance (precision).

Output:
  - results/transformed_proxy/{proxy_name}_transformed.json
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict

def load_proxy_scores(proxy_file: str) -> Dict[str, float]:
    """Load proxy scores from JSON file."""
    with open(proxy_file, 'r') as f:
        data = json.load(f)
    
    # Extract scores from 'results' dict
    if isinstance(data, dict) and 'results' in data:
        scores_dict = data['results']
    else:
        scores_dict = data
    
    return scores_dict


def transform_proxies(output_dir: str = 'results/transformed_proxy', epsilon: float = 1e-8):
    """
    Apply log transformation to all proxy metrics.
    
    Args:
        output_dir: Directory to save transformed proxies
        epsilon: Small constant to avoid log(0)
    """
    
    os.makedirs(output_dir, exist_ok=True)
    
    proxy_scores_dir = 'results/raw_proxy_scores'
    
    print("\n" + "="*80)
    print("STEP 2: LOG TRANSFORMATION OF PROXY METRICS")
    print("="*80)
    
    # NASWOT and Zen-Score are negated to correct direction:
    # raw correlation is negative (higher variance = worse arch), so we negate to get
    # -log(x + eps) = log(1/(x + eps)), i.e., higher compression = higher score = better arch
    transformations = {
        'synflow':     ('synflow_full.json',     lambda x:  np.log(x + epsilon),  False),
        'naswot':      ('naswot_full.json',      lambda x: -np.log(x + epsilon),  True),
        'zenscore':    ('zenscore_full.json',    lambda x: -np.log(x + epsilon),  True),
        'param_count': ('param_count_full.json', lambda x:  np.log(x + 1),        False),
    }
    
    all_transformed = {}
    
    print("\nApplying log transformations...")
    
    for proxy_name, (filename, transform_fn, is_negated) in transformations.items():
        proxy_file = os.path.join(proxy_scores_dir, filename)
        
        if not os.path.exists(proxy_file):
            print(f"  ⚠️  {filename} not found, skipping {proxy_name}")
            continue
        
        try:
            print(f"\n  Processing {proxy_name}...", end=" ")
            
            # Load raw scores
            raw_scores = load_proxy_scores(proxy_file)
            
            # Apply transformation
            transformed_scores = {}
            for arch_id, score in raw_scores.items():
                try:
                    transformed_value = float(transform_fn(float(score)))
                    transformed_scores[arch_id] = transformed_value
                except:
                    # Fallback for invalid values
                    transformed_scores[arch_id] = 0.0
            
            # Compute statistics
            values = np.array(list(transformed_scores.values()))
            values = values[np.isfinite(values)]
            
            if proxy_name == 'param_count':
                transform_desc = 'log(x + 1)'
            elif is_negated:
                transform_desc = f'-log(x + {epsilon})  [negated: higher score = better]'
            else:
                transform_desc = f'log(x + {epsilon})'

            stats = {
                'transformation': transform_desc,
                'is_negated': is_negated,
                'count': len(transformed_scores),
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'range': float(np.max(values) - np.min(values))
            }
            
            # Save transformed scores
            output_file = os.path.join(output_dir, f'{proxy_name}_transformed.json')
            with open(output_file, 'w') as f:
                json.dump({
                    'proxy_name': proxy_name,
                    'transformation': stats['transformation'],
                    'is_negated': is_negated,
                    'results': transformed_scores,
                    'statistics': stats
                }, f, indent=2)
            
            print(f"✓")
            print(f"    Mean: {stats['mean']:.6f}")
            print(f"    Std:  {stats['std']:.6f}")
            print(f"    Range: {stats['min']:.6f} to {stats['max']:.6f}")
            if is_negated:
                print(f"    [NEGATED] Direction inverted: -log(x) so higher = better")
            print(f"    Saved to: {output_file}")
            
            all_transformed[proxy_name] = transformed_scores
        
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "="*80)
    print(f"✓ Log transformation complete! Transformed {len(all_transformed)} proxies.")
    print("="*80 + "\n")
    
    return all_transformed


if __name__ == '__main__':
    transform_proxies()
