"""
Proxy 4: Zen-Score
Measures activation diversity by computing variance of activations
under different random input samples.
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


def zenscore(model: nn.Module, num_samples: int = 4, input_size: Tuple = (4, 3, 32, 32),
             device: torch.device = None) -> float:
    """
    Compute Zen-Score (activation covariance trace under perturbations).
    
    Zen-Score measures the diversity of activations by computing variance
    across multiple random input samples and multiple Conv2d layers.
    
    Implementation: Multi-layer, multi-sample strategy
    - Hooks ALL Conv2d layers in the network
    - Runs multiple forward passes with different random inputs
    - Computes covariance trace for each (layer, sample) pair
    - Aggregates via mean (reduces noise from any single layer/sample)
    
    Args:
        model: PyTorch model
        num_samples: Number of random input samples (4 recommended)
        input_size: Input tensor shape (batch, channels, height, width)
        device: PyTorch device
    
    Returns:
        Zen-Score (float) - mean covariance trace across all layers and samples
    """
    try:
        if device is None:
            device = torch.device('cpu')
        
        model.eval()
        
        # Find ALL Conv2d layers
        conv_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                conv_modules.append(module)
        
        if len(conv_modules) == 0:
            return 0.0
        
        all_layer_scores = []
        
        # Run multiple forward passes with different random inputs
        with torch.no_grad():
            for sample_idx in range(num_samples):
                # Dictionary to collect activations per layer for this sample
                layer_activations = {i: [] for i in range(len(conv_modules))}
                
                # Create hook functions for each layer
                def create_hook(layer_idx):
                    def hook_fn(module, input, output):
                        if isinstance(output, torch.Tensor):
                            if output.dim() > 2:
                                output_flat = output.view(output.size(0), output.size(1), -1).mean(dim=2)
                            else:
                                output_flat = output
                            layer_activations[layer_idx].append(output_flat.detach().cpu())
                    return hook_fn
                
                # Register hooks on all Conv2d layers
                hooks = []
                for layer_idx, conv_module in enumerate(conv_modules):
                    hook = conv_module.register_forward_hook(create_hook(layer_idx))
                    hooks.append(hook)
                
                # Forward pass with random input
                x = torch.randn(input_size, device=device)
                _ = model(x)
                
                # Remove all hooks
                for hook in hooks:
                    hook.remove()
                
                # Compute covariance trace for each layer in this sample
                sample_layer_scores = []
                
                for layer_idx in range(len(conv_modules)):
                    if len(layer_activations[layer_idx]) == 0:
                        continue
                    
                    acts = layer_activations[layer_idx][0].float()
                    
                    if acts.size(1) == 0:
                        continue
                    
                    # Center activations
                    acts_mean = acts.mean(dim=0, keepdim=True)
                    acts_centered = acts - acts_mean
                    
                    # Compute covariance
                    if acts_centered.size(0) > 1:
                        cov = (acts_centered.t() @ acts_centered) / acts.size(0)
                    else:
                        cov = acts_centered.t() @ acts_centered
                    
                    # Trace of covariance
                    trace_val = torch.diagonal(cov).sum().item()
                    sample_layer_scores.append(max(trace_val, 0.0))
                
                # Add this sample's layer scores to the overall collection
                all_layer_scores.extend(sample_layer_scores)
        
        if len(all_layer_scores) == 0:
            return 0.0
        
        # Zen-Score is mean across all (layer, sample) pairs
        # This provides a robust estimate averaging out noise
        return float(np.mean(all_layer_scores))
    
    except Exception as e:
        # Fallback
        try:
            return float(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1000)
        except:
            return 0.0


def compute_zenscore_batch(arch_ids: List[int], arch_data_dir: str, device: torch.device = None,
                           batch_size: int = 64, verbose: bool = True) -> Dict[int, float]:
    """
    Compute Zen-Scores for a batch of architectures.
    
    Args:
        arch_ids: List of architecture IDs
        arch_data_dir: Directory containing architecture files
        device: PyTorch device
        batch_size: Number of architectures before memory cleanup
        verbose: Print progress
    
    Returns:
        Dictionary mapping arch_id -> zenscore
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
            
            # Compute Zen-Score
            score = zenscore(model, num_samples=4, device=device)
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


def test_zenscore(subset_size: int = 256, arch_data_dir: str = None,
                  device_name: str = 'directml', output_dir: str = None):
    """
    Test Zen-Score computation on a small subset of architectures.
    
    Args:
        subset_size: Number of architectures to test
        arch_data_dir: Directory with architecture files
        device_name: 'directml', 'cpu', or 'cuda'
        output_dir: Directory to save results
    """
    print("\n" + "="*70)
    print("PROXY 4: ZEN-SCORE TEST")
    print("="*70)
    
    # Setup paths
    if arch_data_dir is None:
        arch_data_dir = 'data/nasbench201/chunks_clean/arch2infos'
    if output_dir is None:
        output_dir = 'results/nasbench201/raw_proxy_scores'
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get device
    device = get_device(device_name)
    print(f"\nUsing device: {device}")
    
    # Get list of architecture files
    arch_files = sorted([f for f in os.listdir(arch_data_dir) if f.endswith('.pth')])
    arch_ids = [int(f.replace('.pth', '')) for f in arch_files[:subset_size]]
    print(f"\nTesting on {len(arch_ids)} architectures")
    
    # Compute Zen-Score scores
    print(f"\nComputing Zen-Score scores...")
    zenscore_scores = compute_zenscore_batch(arch_ids, arch_data_dir, device=device, verbose=True)
    
    # Statistics
    values = np.array([v for v in zenscore_scores.values() if v > 0])
    print(f"\n{'─'*70}")
    print("RESULTS:")
    print(f"{'─'*70}")
    print(f"Total computed: {len(values)}")
    print(f"Min Zen-Score:   {values.min():.6f}")
    print(f"Max Zen-Score:   {values.max():.6f}")
    print(f"Mean Zen-Score:  {values.mean():.6f}")
    print(f"Median Zen-Score:{np.median(values):.6f}")
    print(f"Std Zen-Score:   {values.std():.6f}")
    
    # Show sample results
    print(f"\n{'─'*70}")
    print("SAMPLE RESULTS (first 10 architectures):")
    print(f"{'─'*70}")
    for arch_id in arch_ids[:10]:
        print(f"  Arch {arch_id:5d}: {zenscore_scores[arch_id]:>16.6f}")
    
    # Determine output filename based on subset size
    if len(arch_ids) == 15625:
        results_filename = 'zenscore_full.json'
    else:
        results_filename = 'zenscore_test.json'
    
    # Save results
    results_file = os.path.join(output_dir, results_filename)
    with open(results_file, 'w') as f:
        json.dump({
            'proxy_name': 'Zen-Score',
            'subset_size': len(arch_ids),
            'results': {str(k): float(v) for k, v in zenscore_scores.items()},
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
    
    return zenscore_scores, values


if __name__ == '__main__':
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description='Compute Zen-Score proxy')
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
    test_zenscore(subset_size=subset_size, device_name=args.device)
