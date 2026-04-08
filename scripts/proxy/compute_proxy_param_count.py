"""
Proxy 1: Parameter Count
Measures total number of trainable parameters in the network.
This is a simple structural metric that doesn't require model inference.
"""
import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add proxy utils to path
sys.path.insert(0, str(Path(__file__).parent))
from proxy_utils import build_nas201_model, count_parameters, get_device


def compute_param_count(arch_str: str, C: int = 16, num_cells: int = 5, device: torch.device = None) -> float:
    """
    Compute parameter count for a given architecture.
    
    Args:
        arch_str: NAS-Bench-201 architecture string
        C: Channel multiplier (default 16 for NAS-Bench-201)
        num_cells: Number of cells
        device: PyTorch device
    
    Returns:
        Parameter count as float
    """
    try:
        model = build_nas201_model(arch_str, C=C, num_cells=num_cells, device=str(device))
        param_count = count_parameters(model)
        return float(param_count)
    except Exception as e:
        print(f"  ⚠ Error computing param count: {e}")
        return 0.0


def compute_proxies_batch(arch_ids: List[int], arch_data_dir: str, device: torch.device = None, 
                          batch_size: int = 64, verbose: bool = True) -> Dict[int, float]:
    """
    Compute parameter count for a batch of architectures.
    
    Args:
        arch_ids: List of architecture IDs to process
        arch_data_dir: Directory containing architecture chunk files
        device: PyTorch device
        batch_size: Batch size (not used for param count, but kept for API consistency)
        verbose: Print progress
    
    Returns:
        Dictionary mapping arch_id -> param_count
    """
    results = {}
    
    for i, arch_id in enumerate(arch_ids):
        # Load architecture from chunk
        chunk_path = os.path.join(arch_data_dir, f"{arch_id}.pth")
        
        try:
            chunk_data = torch.load(chunk_path, weights_only=False)
            arch_idx = list(chunk_data.keys())[0]
            arch_str = chunk_data[arch_idx]['full']['arch_str']
            
            # Compute parameter count
            param_count = compute_param_count(arch_str, device=device)
            results[arch_id] = param_count
            
            if verbose and (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(arch_ids)} architectures")
        
        except Exception as e:
            print(f"  Error loading arch {arch_id}: {e}")
            results[arch_id] = 0.0
    
    return results


def test_param_count(subset_size: int = 256, arch_data_dir: str = None, 
                     device_name: str = 'directml', output_dir: str = None):
    """
    Test parameter count computation on a small subset of architectures.
    
    Args:
        subset_size: Number of architectures to test
        arch_data_dir: Directory with architecture files
        device_name: 'directml', 'cpu', or 'cuda'
        output_dir: Directory to save results
    """
    print("\n" + "="*70)
    print("PROXY 1: PARAMETER COUNT TEST")
    print("="*70)
    
    # Setup paths
    if arch_data_dir is None:
        arch_data_dir = 'data/chunks_clean/arch2infos'
    if output_dir is None:
        output_dir = 'results/proxy_scores'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = get_device(device_name)
    print(f"\nUsing device: {device}")
    
    # Get list of architecture files
    arch_files = sorted([f for f in os.listdir(arch_data_dir) if f.endswith('.pth')])
    arch_ids = [int(f.replace('.pth', '')) for f in arch_files[:subset_size]]
    print(f"\nTesting on {len(arch_ids)} architectures")
    
    # Compute parameter counts
    print(f"\nComputing parameter counts...")
    param_counts = compute_proxies_batch(arch_ids, arch_data_dir, device=device, verbose=True)
    
    # Statistics
    values = np.array([v for v in param_counts.values() if v > 0])
    print(f"\n{'─'*70}")
    print("RESULTS:")
    print(f"{'─'*70}")
    print(f"Total computed: {len(values)}")
    print(f"Min params:    {values.min():,.0f}")
    print(f"Max params:    {values.max():,.0f}")
    print(f"Mean params:   {values.mean():,.0f}")
    print(f"Median params: {np.median(values):,.0f}")
    print(f"Std params:    {values.std():,.0f}")
    
    # Show sample results
    print(f"\n{'─'*70}")
    print("SAMPLE RESULTS (first 10 architectures):")
    print(f"{'─'*70}")
    for arch_id in arch_ids[:10]:
        print(f"  Arch {arch_id:5d}: {param_counts[arch_id]:>10,.0f} parameters")
    
    # Determine output filename based on subset size
    if len(arch_ids) == 15625:
        results_filename = 'param_count_full.json'
    else:
        results_filename = 'param_count_test.json'
    
    # Save results
    results_file = os.path.join(output_dir, results_filename)
    with open(results_file, 'w') as f:
        json.dump({
            'proxy_name': 'Parameter Count',
            'subset_size': len(arch_ids),
            'results': {str(k): float(v) for k, v in param_counts.items()},
            'statistics': {
                'min': float(values.min()),
                'max': float(values.max()),
                'mean': float(values.mean()),
                'median': float(np.median(values)),
                'std': float(values.std()),
                'count': len(values)
            }
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {results_file}")
    print(f"\n{'='*70}")
    
    return param_counts, values


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute Parameter Count proxy')
    parser.add_argument('--subset', type=int, default=256, 
                        help='Number of architectures to process (default: 256)')
    parser.add_argument('--full', action='store_true',
                        help='Process all 15,625 architectures')
    parser.add_argument('--device', default='directml',
                        help='Device to use: directml, cpu, cuda (default: directml)')
    
    args = parser.parse_args()
    
    # Determine subset size
    subset_size = 15625 if args.full else args.subset
    
    # Run test
    test_param_count(subset_size=subset_size, device_name=args.device)
