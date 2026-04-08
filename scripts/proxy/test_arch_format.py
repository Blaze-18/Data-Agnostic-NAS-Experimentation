"""
Quick inspection of architecture string format from one chunk file.
"""
import torch
import json

chunk_file = 'data/chunks_clean/arch2infos/0.pth'

try:
    data = torch.load(chunk_file, weights_only=False)
    
    # Print top-level keys
    print("Top-level keys:", list(data.keys()))
    
    # Get arch_idx
    arch_idx = list(data.keys())[0]
    arch_data = data[arch_idx]
    
    print(f"\nSample Architecture ID: {arch_idx}")
    print(f"Keys in arch data: {list(arch_data.keys())}")
    
    if 'arch_str' in arch_data:
        print(f"\nArchitecture String: {arch_data['arch_str']}")
        print(f"Type: {type(arch_data['arch_str'])}")
    
    if 'full' in arch_data:
        full = arch_data['full']
        print(f"\nFull keys: {list(full.keys())}")
        if 'all_results' in full:
            results_keys = list(full['all_results'].keys())
            print(f"Number of (dataset, seed) keys: {len(results_keys)}")
            print(f"Sample keys: {results_keys[:3]}")
            
            # Check one result
            first_key = results_keys[0]
            result = full['all_results'][first_key]
            print(f"\nSample result for {first_key}:")
            print(f"  Keys: {list(result.keys())}")
            if 'train_acc1es' in result:
                train_acc_epochs = result['train_acc1es']
                print(f"  train_acc1es (epochs): {len(train_acc_epochs)} epochs")
                print(f"  Final epoch (199) accuracy: {train_acc_epochs.get(199, 'N/A')}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
