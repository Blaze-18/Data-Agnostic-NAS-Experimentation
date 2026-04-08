"""
Convert a .pth file (top-level dict) into an HDF5 file, writing each tensor/array as a dataset
and deleting entries from memory as they are written to reduce RAM usage.

Usage:
    python scripts/convert_pth_to_h5.py --input data/raw/NAS-Bench-201-v1_0-e61699.pth --out data/processed/dataset.h5
"""
import argparse
import os
import torch
import h5py
import numpy as np
import pickle


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--out', '-o', required=True)
    args = p.parse_args()

    inp = args.input
    out = args.out

    if not os.path.exists(inp):
        print('Input file not found:', inp)
        return

    print('Loading input (this happens on local machine):', inp)
    try:
        import numpy as _np
        with torch.serialization.safe_globals([_np.core.multiarray.scalar]):
            data = torch.load(inp, map_location='cpu', weights_only=False)
    except Exception as e:
        print('safe_globals loading failed, retrying with weights_only=False. Error:', e)
        data = torch.load(inp, map_location='cpu', weights_only=False)

    print('Loaded. Type:', type(data))
    ensure_dir = lambda p: os.makedirs(os.path.dirname(p), exist_ok=True)
    ensure_dir(out)

    with h5py.File(out, 'w') as hf:
        if isinstance(data, dict):
            for i, (k, v) in enumerate(list(data.items())):
                name = str(k)
                print(f'Writing key {i}/{len(data)}: {name}')
                try:
                    if torch.is_tensor(v):
                        arr = v.cpu().numpy()
                        hf.create_dataset(name, data=arr, compression='gzip')
                    else:
                        # Fallback: store pickled bytes
                        hf.create_dataset(name, data=np.void(pickle.dumps(v)))
                except Exception as e:
                    print('Error writing key', name, e)
                # delete to free memory
                try:
                    del data[k]
                except Exception:
                    pass
        elif isinstance(data, (list, tuple)):
            grp = hf.create_group('list')
            for i, item in enumerate(data):
                name = f'item_{i}'
                print(f'Writing item {i}/{len(data)}')
                try:
                    if torch.is_tensor(item):
                        arr = item.cpu().numpy()
                        grp.create_dataset(name, data=arr, compression='gzip')
                    else:
                        grp.create_dataset(name, data=np.void(pickle.dumps(item)))
                except Exception as e:
                    print('Error writing item', i, e)
                # free memory
                try:
                    del data[i]
                except Exception:
                    pass
        else:
            # Store as a single dataset
            print('Unknown top-level type, storing pickled object')
            hf.create_dataset('obj', data=np.void(pickle.dumps(data)))

    print('Conversion complete:', out)

if __name__ == '__main__':
    main()
