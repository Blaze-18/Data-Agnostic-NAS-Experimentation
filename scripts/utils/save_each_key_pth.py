"""
Save each top-level key from a .pth (dict) into its own .pth file safely (atomic rename).
This avoids creating very large combined pickle objects and reduces memory pressure.

Usage:
    python scripts/save_each_key_pth.py --input data/raw/NAS-Bench-201-v1_0-e61699.pth --outdir data/chunks_clean
"""
import argparse
import os
import torch
import tempfile
from pathlib import Path


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def safe_save(obj, path):
    # write to temp file then rename for atomicity
    path = Path(path)
    d = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=str(d), prefix='.tmp_', suffix='.pth')
    os.close(fd)
    # Use forward slashes for torch.save compatibility
    tmp = str(tmp_path).replace('\\', '/')
    try:
        torch.save(obj, tmp)
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except Exception:
                pass


def save_nested(key, obj, outdir):
    """For large nested structures, save sub-items individually to avoid MemoryError."""
    if isinstance(obj, dict):
        # Create subdir for this key's items
        subdir = Path(outdir) / str(key)
        subdir.mkdir(exist_ok=True, parents=True)
        for i, (k, v) in enumerate(list(obj.items())):
            try:
                fname = subdir / f'{str(k)}.pth'
                safe_save({k: v}, str(fname))
                print(f'  Saved sub-item {i}: {k}')
            except MemoryError:
                print(f'  MemoryError on sub-item {k}; skipping')
            except Exception as e:
                print(f'  Error saving {k}: {e}')
    elif isinstance(obj, (list, tuple)):
        # Create subdir for this key's items
        subdir = Path(outdir) / str(key)
        subdir.mkdir(exist_ok=True, parents=True)
        for i, item in enumerate(list(obj)):
            try:
                fname = subdir / f'item_{i}.pth'
                safe_save([item], str(fname))
                print(f'  Saved sub-item {i}')
            except MemoryError:
                print(f'  MemoryError on item {i}; skipping')
            except Exception as e:
                print(f'  Error saving item {i}: {e}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--outdir', '-o', required=True)
    args = p.parse_args()

    inp = args.input
    outdir = args.outdir
    ensure_dir(outdir)

    if not os.path.exists(inp):
        print('Input file not found:', inp)
        return

    print('Loading input (trusted source):', inp)
    try:
        import numpy as _np
        with torch.serialization.safe_globals([_np.core.multiarray.scalar]):
            data = torch.load(inp, map_location='cpu', weights_only=False)
    except Exception as e:
        print('safe_globals failed, retrying with weights_only=False. Error:', e)
        data = torch.load(inp, map_location='cpu', weights_only=False)

    if not isinstance(data, dict):
        print('Top-level object is not a dict; saving single file instead')
        safe_save(data, os.path.join(outdir, 'obj_0.pth'))
        print('Saved single object to', outdir)
        return

    print('Top-level keys:', len(data))
    for i, key in enumerate(list(data.keys())):
        print(f'Saving key {i+1}/{len(data)}: {key}')
        obj = {key: data[key]}
        fname = os.path.join(outdir, f'{str(key)}.pth')
        # sanitize filename
        safe_name = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in str(key))
        fname = os.path.join(outdir, safe_name + '.pth')
        
        # Try to save as single file; fallback to nested structure if MemoryError
        try:
            safe_save(obj, fname)
            print(f'  Saved successfully to {safe_name}.pth')
        except MemoryError:
            print(f'  MemoryError: decomposing {key} into sub-items')
            save_nested(safe_name, data[key], outdir)
        except Exception as e:
            print(f'  Error saving {key}: {e}')
        
        # free memory
        try:
            del data[key]
        except Exception:
            pass

    print('All keys saved to', outdir)

if __name__ == '__main__':
    main()
