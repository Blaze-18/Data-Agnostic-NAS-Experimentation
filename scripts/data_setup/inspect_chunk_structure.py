#!/usr/bin/env python3
"""
Quick inspection of .pth chunk file structure to debug extraction.
"""
import torch
from pathlib import Path
import json

root = Path("data/chunks_clean/arch2infos")
files = sorted(root.glob("*.pth"))[:3]  # Inspect first 3 files

print(f"Inspecting {len(files)} sample chunk files...\n")

for f in files:
    print("=" * 80)
    print(f"File: {f.name}")
    print("=" * 80)
    
    try:
        obj = torch.load(f, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  FAILED TO LOAD: {e}")
        continue
    
    print(f"Type: {type(obj)}")
    
    if isinstance(obj, dict):
        print(f"Top-level keys: {list(obj.keys())}")
        
        # Go one level deep
        for top_k, top_v in obj.items():
            print(f"\n  Key '{top_k}' -> {type(top_v).__name__}")
            if isinstance(top_v, dict):
                print(f"    Nested keys: {list(top_v.keys())}")
                
                # Go into 'full' if it exists
                if 'full' in top_v:
                    full_data = top_v['full']
                    print(f"    Inside 'full' ({type(full_data).__name__}):")
                    if isinstance(full_data, dict):
                        print(f"      Keys: {list(full_data.keys())}")
                        
                        # Check all_results
                        if 'all_results' in full_data:
                            ar = full_data['all_results']
                            print(f"      'all_results' type: {type(ar).__name__}")
                            if isinstance(ar, dict):
                                print(f"        Keys in all_results: {list(ar.keys())[:20]}")
                                for ar_k in list(ar.keys())[:5]:
                                    ar_v = ar[ar_k]
                                    if isinstance(ar_v, dict):
                                        print(f"          {ar_k}: {type(ar_v).__name__} with keys {list(ar_v.keys())[:5]}")
                                        # Look inside
                                        for sub_k in list(ar_v.keys())[:3]:
                                            sub_v = ar_v[sub_k]
                                            if isinstance(sub_v, (int, float, str)):
                                                print(f"            {sub_k}: {type(sub_v).__name__} = {str(sub_v)[:80]}")
                                            elif isinstance(sub_v, dict):
                                                print(f"            {sub_k}: {type(sub_v).__name__} with {len(sub_v)} items")
                                                # Look into train_acc1es, train_acc5es
                                                for acc_k in list(sub_v.keys())[:3]:
                                                    acc_v = sub_v[acc_k]
                                                    print(f"              {acc_k}: {acc_v} ({type(acc_v).__name__})")
                                    else:
                                        print(f"          {ar_k}: {type(ar_v).__name__} = {str(ar_v)[:80]}")
    
    elif isinstance(obj, (list, tuple)):
        print(f"Length: {len(obj)}")
        print(f"First few items: {obj[:3]}")
        
    elif isinstance(obj, torch.Tensor):
        print(f"Shape: {obj.shape}, dtype: {obj.dtype}")
    
    print()
