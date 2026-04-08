#!/usr/bin/env python3
"""
Stage 1: Load chunks, extract proxies + size features, validate, apply transformations.

Loads: data/chunks_clean/arch2infos/*.pth
Saves: 
  - data/processed/features.npy        (N x d PCA-whitened residuals)
  - data/processed/labels.npy          (N,) ground-truth accuracies
  - data/processed/transforms.pkl      (transformation params)
  - data/processed/validation_log.json (preprocessing diagnostics)

Run from repo root:
    python scripts/preprocess_locally.py
"""
import json
import pickle
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def extract_from_obj(obj):
    """
    Extract from NAS-Bench-201 chunk structure:
    {arch_id: {'full': {'all_results': {(dataset, seed): {'train_acc1es': {...}}}}}}
    
    Returns (arch_index, label, has_data).
    """
    if not isinstance(obj, dict) or len(obj) != 1:
        return None, None, False
    
    # Get the single arch_id and its data
    arch_id, arch_data = list(obj.items())[0]
    
    if not isinstance(arch_data, dict) or 'full' not in arch_data:
        return None, None, False
    
    full_data = arch_data['full']
    if not isinstance(full_data, dict):
        return None, None, False
    
    # Extract arch_index
    arch_index = full_data.get('arch_index', arch_id)
    
    # Extract label: use final accuracy from CIFAR-10 validation (if available)
    # all_results is {(dataset, seed): {name, train_acc1es: {epoch: acc}, ...}}
    all_results = full_data.get('all_results', {})
    if not isinstance(all_results, dict):
        return arch_index, None, False
    
    # Prefer cifar10-valid as primary label source
    label = None
    for (dataset, seed), results in all_results.items():
        if isinstance(results, dict) and 'train_acc1es' in results:
            accs = results['train_acc1es']
            if isinstance(accs, dict) and len(accs) > 0:
                # Get final epoch (last key or key 199)
                final_acc = accs.get(199) or accs.get(max(accs.keys())) if accs else None
                if final_acc is not None:
                    label = float(final_acc)
                    # Prefer cifar10-valid
                    if dataset == 'cifar10-valid':
                        break
    
    return arch_index, label, label is not None


def main():
    root = Path("data/chunks_clean/arch2infos")
    outdir = Path("data/processed")
    outdir.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        print(f"ERROR: Chunk directory not found: {root}")
        return

    files = sorted(root.glob("*.pth"))
    if len(files) == 0:
        print(f"ERROR: No .pth files in {root}")
        return

    print(f"Found {len(files)} chunk files. Starting extraction...")

    records = []
    validation_log = {
        "total_files": len(files),
        "loaded_successfully": 0,
        "load_failed": 0,
        "records_before_validation": 0,
        "records_after_validation": 0,
        "validation_failures": {},
        "has_proxy_fields_count": 0,
        "missing_info": {},
    }

    # Load and extract
    for i, p in enumerate(files):
        if (i + 1) % 2000 == 0:
            print(f"  Processed {i+1}/{len(files)}...")
        
        try:
            obj = torch.load(p, map_location="cpu", weights_only=False)
            validation_log["loaded_successfully"] += 1
        except Exception as e:
            validation_log["load_failed"] += 1
            continue

        arch_index, label, has_data = extract_from_obj(obj)
        
        if not has_data:
            validation_log["validation_failures"]["no_label"] = validation_log["validation_failures"].get("no_label", 0) + 1
            continue
        
        if label is None or np.isnan(label):
            validation_log["validation_failures"]["label_nan"] = validation_log["validation_failures"].get("label_nan", 0) + 1
            continue
        
        # Sanity check: label should be 0-100 (percentages in NAS-Bench-201)
        if label < 0 or label > 100:
            validation_log["validation_failures"][f"label_out_of_range"] = validation_log["validation_failures"].get("label_out_of_range", 0) + 1
            continue
        
        rec = {"file": str(p.name), "arch_index": arch_index, "label": label}
        records.append(rec)

    validation_log["records_before_validation"] = len(records)
    df = pd.DataFrame(records)
    print(f"Extracted {len(df)} records with valid labels.")

    if len(df) == 0:
        print("ERROR: No valid records extracted. Aborting.")
        return

    validation_log["records_after_validation"] = len(df)

    # For now, save labels only (no proxies found in chunks)
    # Proxies will be computed separately
    labels = df["label"].values.astype(np.float32)
    arch_indices = df["arch_index"].values

    np.save(outdir / "labels.npy", labels)
    np.save(outdir / "arch_indices.npy", arch_indices)

    # Create minimal transforms dict for now
    transforms = {
        "note": "Chunk files contain no precomputed proxies. Extract labels only.",
        "arch_indices": arch_indices.tolist(),
        "n_samples": len(labels),
        "label_min": float(labels.min()),
        "label_max": float(labels.max()),
        "label_mean": float(labels.mean()),
        "label_std": float(labels.std()),
    }
    with open(outdir / "transforms.pkl", "wb") as f:
        pickle.dump(transforms, f)

    # Create placeholder features (1D - just the arch_index for now)
    features = arch_indices.astype(np.float32).reshape(-1, 1)
    np.save(outdir / "features.npy", features)

    validation_log.update({
        "labels_shape": labels.shape,
        "labels_min": float(labels.min()),
        "labels_max": float(labels.max()),
        "labels_mean": float(labels.mean()),
        "labels_std": float(labels.std()),
        "note": "No proxies in chunk files. Placeholder features created. TODO: Compute actual proxies.",
    })

    with open(outdir / "validation_log.json", "w") as f:
        json.dump(validation_log, f, indent=2)

    print("\n" + "="*60)
    print("LABEL EXTRACTION COMPLETE")
    print("="*60)
    print(f"Labels shape: {labels.shape}")
    print(f"Label range: [{labels.min():.2f}, {labels.max():.2f}]")
    print(f"Label mean ± std: {labels.mean():.2f} ± {labels.std():.2f}")
    print(f"Files saved to: {outdir}")
    print(f"  - labels.npy ({labels.shape})")
    print(f"  - arch_indices.npy ({arch_indices.shape})")
    print(f"  - features.npy (placeholder: arch_indices)")
    print(f"  - transforms.pkl")
    print(f"  - validation_log.json")
    print(f"\n⚠️  NOTE: No proxies extracted from chunks.")
    print(f"Next steps:")
    print(f"  1. Compute structural proxies (SynFlow, Zen, etc.)")
    print(f"  2. Update features.npy + transforms.pkl with proxy data")
    print(f"  3. Run: python scripts/visualize_preprocessing.py")
    print("="*60)


if __name__ == "__main__":
    main()

