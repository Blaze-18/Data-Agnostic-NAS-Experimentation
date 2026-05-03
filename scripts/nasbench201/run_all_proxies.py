"""
Master Proxy Computation Orchestrator
Runs all 4 proxies sequentially or independently with unified logging and progress tracking.
"""
import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple


def run_proxy_suite(subset_size: int = 256, device: str = 'directml', 
                    proxies: List[str] = None, output_dir: str = None,
                    full_dataset: bool = False) -> Dict:
    """
    Run proxy computation suite on architectures.
    
    Args:
        subset_size: Number of architectures to test (256 for validation, 15625 for full)
        device: 'directml', 'cpu', or 'cuda'
        proxies: List of proxies to run. Default: ['param_count', 'synflow', 'naswot', 'zenscore']
        output_dir: Directory to save results
        full_dataset: If True, run on all 15,625 architectures
    
    Returns:
        Dictionary with results from all proxies
    """
    if proxies is None:
        proxies = ['param_count', 'synflow', 'naswot', 'zenscore']
    
    if output_dir is None:
        output_dir = 'results/nasbench201/raw_proxy_scores'
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if full_dataset:
        subset_size = 15625
    
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  ZERO-COST PROXY COMPUTATION SUITE".center(78) + "█")
    print("█" + f"  Dataset: {subset_size:,} architectures | Device: {device.upper()}".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80 + "\n")
    
    results = {}
    all_results_file = os.path.join(output_dir, 'all_proxies_results.json')
    
    start_time = time.time()
    
    # Mapping of proxy names to script filenames
    proxy_scripts = {
        'param_count': 'compute_proxy_param_count.py',
        'synflow': 'compute_proxy_synflow.py',
        'naswot': 'compute_proxy_naswot.py',
        'zenscore': 'compute_proxy_zenscore.py',
    }
    
    proxy_names = {
        'param_count': 'Parameter Count',
        'synflow': 'SynFlow',
        'naswot': 'NASWOT',
        'zenscore': 'Zen-Score',
    }
    
    proxy_times = {}
    
    # Scripts are in the same directory as this script
    scripts_dir = str(Path(__file__).parent)
    
    for i, proxy_key in enumerate(proxies):
        if proxy_key not in proxy_scripts:
            print(f"⚠️ Unknown proxy: {proxy_key}")
            continue
        
        proxy_name = proxy_names[proxy_key]
        script_path = os.path.join(scripts_dir, proxy_scripts[proxy_key])
        
        print(f"\n[{i+1}/{len(proxies)}] Running {proxy_name}...")
        print("─" * 80)
        
        if not os.path.exists(script_path):
            print(f"❌ Script not found: {script_path}")
            results[proxy_key] = {
                'status': 'error',
                'error': f'Script not found: {script_path}',
                'time_sec': 0
            }
            continue
        
        proxy_start = time.time()
        
        try:
            # Run the proxy script as subprocess
            cmd = [sys.executable, script_path]
            
            # Add arguments for full dataset run
            if full_dataset:
                cmd.append('--full')
            else:
                cmd.append('--subset')
                cmd.append(str(subset_size))
            
            cmd.append('--device')
            cmd.append(device)
            
            print(f"   Command: {' '.join(cmd)}")
            print(f"   Subset size: {subset_size:,} architectures")
            print(f"   Device: {device}")
            print(f"   Output: {output_dir}")
            print()
            
            result = subprocess.run(
                cmd,
                cwd=str(Path(__file__).parent.parent.parent),  # Run from workspace root
                capture_output=False,
                timeout=3600  # 1 hour timeout per proxy
            )
            
            proxy_time = time.time() - proxy_start
            
            if result.returncode == 0:
                proxy_times[proxy_key] = proxy_time
                results[proxy_key] = {
                    'status': 'success',
                    'time_sec': proxy_time,
                    'script': script_path
                }
                print(f"\n✅ {proxy_name} completed in {proxy_time:.1f} seconds ({proxy_time/60:.1f} min)")
            else:
                results[proxy_key] = {
                    'status': 'error',
                    'error': f'Exit code {result.returncode}',
                    'time_sec': proxy_time
                }
                print(f"\n❌ {proxy_name} failed with exit code {result.returncode}")
        
        except subprocess.TimeoutExpired:
            proxy_time = time.time() - proxy_start
            results[proxy_key] = {
                'status': 'error',
                'error': 'Timeout (exceeded 1 hour)',
                'time_sec': proxy_time
            }
            print(f"\n⏱️ {proxy_name} timed out after {proxy_time:.1f} seconds")
        
        except Exception as e:
            proxy_time = time.time() - proxy_start
            results[proxy_key] = {
                'status': 'error',
                'error': str(e),
                'time_sec': proxy_time
            }
            print(f"\n❌ {proxy_name} failed: {e}")
    
    total_time = time.time() - start_time
    
    # Save aggregated results
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'subset_size': subset_size,
            'device': device,
            'proxies_run': proxies
        },
        'results_summary': {
            proxy: {k: v for k, v in data.items() if k != 'results'}
            for proxy, data in results.items()
        },
        'timing': {
            'individual': proxy_times,
            'total_sec': total_time
        }
    }
    
    with open(all_results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print summary
    print("\n" + "─" * 80)
    print("PROXY SUITE SUMMARY")
    print("─" * 80)
    
    for proxy_key, proxy_data in results.items():
        proxy_name = proxy_names[proxy_key]
        status = proxy_data['status']
        time_sec = proxy_data.get('time_sec', 0)
        
        if status == 'success':
            minutes = time_sec / 60
            if minutes < 1:
                time_str = f"{time_sec:.0f}s"
            else:
                time_str = f"{minutes:.1f}m"
            print(f"✅ {proxy_name:20s} |  {time_str:>6}")
        else:
            print(f"❌ {proxy_name:20s} | ERROR: {proxy_data.get('error', 'Unknown')}")
    
    print("─" * 80)
    total_minutes = total_time / 60
    if total_minutes < 1:
        total_str = f"{total_time:.0f} seconds"
    else:
        total_str = f"{total_minutes:.1f} minutes"
    
    print(f"Total runtime: {total_str}")
    print(f"Results saved to: {all_results_file}")
    print("─" * 80 + "\n")
    
    return results


def print_help():
    """Print usage information."""
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║     ZERO-COST PROXY COMPUTATION SUITE - Usage Guide                        ║
╚════════════════════════════════════════════════════════════════════════════╝

QUICK START:
  python run_all_proxies.py                    # Validate on 256 archs (~30 min)
  python run_all_proxies.py --full             # Full 15,625 archs (~2 hours)
  python run_all_proxies.py --full --fast      # Fast track: param_count+synflow (~35 min)

INDIVIDUAL SCRIPTS:
  python compute_proxy_param_count.py           # Fast: ~5 min (full) / ~20 sec (256)
  python compute_proxy_synflow.py               # Medium: ~30 min (full) / ~8 min (256)
  python compute_proxy_naswot.py                # Medium: ~20 min (full) / ~4 min (256)
  python compute_proxy_zenscore.py              # Medium: ~20 min (full) / ~4 min (256)

MASTER SCRIPT OPTIONS:
  --subset N              Test on N architectures (default: 256)
  --full                  Run on all 15,625 architectures
  --fast                  Fast track: param_count + synflow only
  --device DEVICE         Device: 'directml', 'cpu', or 'cuda' (default: directml)
  --proxies P1 P2 ...     Run only specified proxies (default: all 4)
  --help                  Show this message

EXAMPLES:
  # Validate on 256 architectures (all 4 proxies, ~30 min)
  python run_all_proxies.py --subset 256
  
  # Full dataset with all proxies (~2 hours)
  python run_all_proxies.py --full
  
  # Fast production run (~35 min, recommended)
  python run_all_proxies.py --full --fast
  
  # Only parameter count (sanity check)
  python run_all_proxies.py --proxies param_count
  
  # Use CPU instead of GPU
  python run_all_proxies.py --subset 256 --device cpu
  
  # Custom selection
  python run_all_proxies.py --full --proxies param_count synflow naswot

EXPECTED RESULTS:
  ✅ Parameter Count:  256/256 or 15,625/15,625 (100% coverage)
  ✅ SynFlow:          256/256 or 15,625/15,625 (100% coverage)
  ⚠️  NASWOT:          66/256 or ~4,000/15,625 (26% coverage - normal)
  ⚠️  Zen-Score:       66/256 or ~4,000/15,625 (26% coverage - normal)

TIMING ESTIMATE:
  Parameter Count:    5 min (full) / 20 sec (256)
  SynFlow:           30 min (full) / 8 min (256)
  NASWOT:            20 min (full) / 4 min (256)
  Zen-Score:         20 min (full) / 4 min (256)
  ─────────────────────────────────────────────
  Total (all):       75 min (full) / 16 min (256)
  Fast track:        35 min (full) / 8 min (256)

NEXT STEPS:
  1. Run this script to compute proxies
  2. Check scripts/proxy/*_results.json for outputs
  3. Run bias_disentanglement.py (removes size bias)
  4. Run visualize_preprocessing.py (generates plots)
  5. Upload to Google Drive and train MLP in Colab

For detailed documentation, see:
  - README.md: Technical guide and theory
  - EXECUTION_ROADMAP.md: Phase-by-phase timeline
  - TEST_RESULTS_SUMMARY.md: Validation results
  - INDEX.md: Complete file manifest
    """)


def print_device_info():
    """Print device information."""
    import torch
    print("\n📊 Device Information:")
    print("─" * 80)
    try:
        # Try DirectML
        device = torch.device("privateuseone:0")
        print(f"✅ DirectML available: {device}")
    except:
        print("❌ DirectML not available")
    
    try:
        # Try CUDA
        if torch.cuda.is_available():
            print(f"✅ CUDA available: cuda (GPU: {torch.cuda.get_device_name(0)})")
        else:
            print("❌ CUDA not available")
    except:
        pass
    
    print(f"✅ CPU available: cpu ({torch.get_num_threads()} threads)")
    print("─" * 80 + "\n")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Zero-Cost Proxy Computation Suite',
        add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--subset', type=int, default=256, 
                        help='Number of architectures to test (default: 256)')
    parser.add_argument('--full', action='store_true', 
                        help='Run on all 15,625 architectures')
    parser.add_argument('--fast', action='store_true',
                        help='Fast track: param_count + synflow only')
    parser.add_argument('--device', default='directml', 
                        help='Device: directml, cpu, cuda (default: directml)')
    parser.add_argument('--proxies', nargs='+', 
                        help='Proxies to run (default: all 4)')
    parser.add_argument('--info', action='store_true',
                        help='Show device information and exit')
    parser.add_argument('--help', '-h', action='store_true', 
                        help='Show this help message and exit')
    
    args = parser.parse_args()
    
    if args.help:
        print_help()
        sys.exit(0)
    
    if args.info:
        print_device_info()
        sys.exit(0)
    
    # Determine subset size
    subset_size = 15625 if args.full else args.subset
    
    # Determine proxies
    if args.proxies:
        proxies = args.proxies
    elif args.fast:
        proxies = ['param_count', 'synflow']
        print("📌 Fast track mode: Running param_count + synflow only\n")
    else:
        proxies = ['param_count', 'synflow', 'naswot', 'zenscore']
    
    # Run suite
    try:
        results = run_proxy_suite(
            subset_size=subset_size,
            device=args.device,
            proxies=proxies,
            output_dir='results/nasbench201/raw_proxy_scores',
            full_dataset=args.full
        )
        
        # Check results
        success_count = sum(1 for r in results.values() if r['status'] == 'success')
        error_count = sum(1 for r in results.values() if r['status'] == 'error')
        
        print("\n" + "=" * 80)
        if error_count == 0:
            print("✅ PROXY SUITE COMPLETED SUCCESSFULLY")
        else:
            print(f"⚠️  PROXY SUITE COMPLETED WITH {error_count} ERROR(S)")
        print("=" * 80)
        
        print(f"\n📊 Summary:")
        print(f"  ✅ Successful: {success_count}/{len(results)}")
        print(f"  ❌ Failed: {error_count}/{len(results)}")
        print(f"\n📁 Results saved to:")
        print(f"  scripts/proxy/all_proxies_results.json")
        print(f"  scripts/proxy/*_results.json")
        print(f"\n📚 Next steps:")
        print(f"  1. Review test results")
        print(f"  2. Run bias_disentanglement.py")
        print(f"  3. Run visualize_preprocessing.py")
        print(f"  4. Upload to Google Drive")
        print(f"  5. Train MLP in Colab")
        print()
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error running proxy suite: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
