"""
Proxy 2: SynFlow
Measures gradient flow (synaptic salience) through the network.
Computes the gradient magnitude of a random forward/backward pass.
"""
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Add proxy utils to path
sys.path.insert(0, str(Path(__file__).parent))
from proxy_utils import build_nas201_model, get_device


def synflow_score(model: nn.Module, input_size: Tuple = (1, 3, 32, 32), 
                  device: torch.device = None) -> float:
    """
    Compute SynFlow score for a model.
    
    SynFlow measures the sum of absolute element-wise layer-wise gradient-weight products,
    indicating how much information flows through the network.
    
    Args:
        model: PyTorch model
        input_size: Input tensor shape
        device: PyTorch device
    
    Returns:
        SynFlow score (float)
    """
    try:
        if device is None:
            device = torch.device('cpu')
        
        # Create random input and target
        x = torch.randn(input_size, device=device)
        
        # Forward pass
        model.train()
        model.zero_grad()
        logits = model(x)
        
        # Random labels for backward (to ensure gradient flow)
        targets = torch.randint(0, 10, (input_size[0],), device=device)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, targets)
        
        # Backward pass
        loss.backward()
        
        # Compute SynFlow: sum of |grad * weight| for all parameters
        synflow_val = 0.0
        for param in model.parameters():
            if param.grad is not None:
                synflow_val += (param.grad * param).abs().sum().item()
        
        return float(synflow_val)
    
    except Exception as e:
        print(f"  ⚠ Error computing SynFlow: {e}")
        return 0.0


def compute_synflow_batch(arch_ids: List[int], arch_data_dir: str, device: torch.device = None,
                          batch_size: int = 64, verbose: bool = True) -> Dict[int, float]:
    """
    Compute SynFlow scores for a batch of architectures.
    
    Args:
        arch_ids: List of architecture IDs
        arch_data_dir: Directory containing architecture files
        device: PyTorch device
        batch_size: Number of architectures to process before clearing GPU memory
        verbose: Print progress
    
    Returns:
        Dictionary mapping arch_id -> synflow_score
    """
    results = {}
    
    for i, arch_id in enumerate(arch_ids):
        # Load architecture
        chunk_path = os.path.join(arch_data_dir, f"{arch_id}.pth")
        
        try:
            chunk_data = torch.load(chunk_path, weights_only=False)
            arch_idx = list(chunk_data.keys())[0]
            arch_str = chunk_data[arch_idx]['full']['arch_str']
            
            # Build model
            model = build_nas201_model(arch_str, device=str(device))
            
            # Compute SynFlow
            score = synflow_score(model, device=device)
            results[arch_id] = score
            
            # Cleanup
            del model
            if device and hasattr(device, 'empty_cache'):
                device.empty_cache()
            
            if verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(arch_ids)} architectures")
        
        except Exception as e:
            print(f"  Error processing arch {arch_id}: {e}")
            results[arch_id] = 0.0
    
    return results


def test_synflow(subset_size: int = 256, arch_data_dir: str = None,
                 device_name: str = 'directml', output_dir: str = None):
    """
    Test SynFlow computation on a small subset of architectures.
    
    Args:
        subset_size: Number of architectures to test
        arch_data_dir: Directory with architecture files
        device_name: 'directml', 'cpu', or 'cuda'
        output_dir: Directory to save results
    """
    print("\n" + "="*70)
    print("PROXY 2: SYNFLOW TEST")
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
    
    # Compute SynFlow scores
    print(f"\nComputing SynFlow scores...")
    synflow_scores = compute_synflow_batch(arch_ids, arch_data_dir, device=device, batch_size=8, verbose=True)
    
    # Statistics
    values = np.array([v for v in synflow_scores.values() if v > 0])
    print(f"\n{'─'*70}")
    print("RESULTS:")
    print(f"{'─'*70}")
    print(f"Total computed: {len(values)}")
    print(f"Min SynFlow:    {values.min():.6e}")
    print(f"Max SynFlow:    {values.max():.6e}")
    print(f"Mean SynFlow:   {values.mean():.6e}")
    print(f"Median SynFlow: {np.median(values):.6e}")
    print(f"Std SynFlow:    {values.std():.6e}")
    
    # Show sample results
    print(f"\n{'─'*70}")
    print("SAMPLE RESULTS (first 10 architectures):")
    print(f"{'─'*70}")
    for arch_id in arch_ids[:10]:
        print(f"  Arch {arch_id:5d}: {synflow_scores[arch_id]:>16.6e}")
    
    # Determine output filename based on subset size
    if len(arch_ids) == 15625:
        results_filename = 'synflow_full.json'
    else:
        results_filename = 'synflow_test.json'
    
    # Save results
    results_file = os.path.join(output_dir, results_filename)
    with open(results_file, 'w') as f:
        json.dump({
            'proxy_name': 'SynFlow',
            'subset_size': len(arch_ids),
            'results': {str(k): float(v) for k, v in synflow_scores.items()},
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
    
    return synflow_scores, values


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute SynFlow proxy')
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
    test_synflow(subset_size=subset_size, device_name=args.device)
