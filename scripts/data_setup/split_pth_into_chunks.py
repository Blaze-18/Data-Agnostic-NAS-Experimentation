"""
Split a large .pth (PyTorch) file into multiple smaller .pth chunk files.
Usage (after activating your venv):
    python split_pth_into_chunks.py --input Dataset/NAS-Bench-201-v1_0-e61699.pth --outdir Dataset/chunks --max-bytes 200000000

This script groups dict entries until approximately `max_bytes` is reached, then writes a chunk file.
If the .pth is a list/tuple, it will write each element as an individual item and group similarly.
"""
import argparse
import os
import torch
import pickle


def estimate_size(obj):
    try:
        return len(pickle.dumps(obj))
    except Exception:
        return 0


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def split_dict(data, outdir, max_bytes):
    chunk = {}
    chunk_size = 0
    idx = 0

    # Iterate over a static list of keys so we can delete items after saving to free RAM
    keys = list(data.keys())
    for key in keys:
        val = data[key]
        s = estimate_size(val)

        # If a single item > max_bytes, save it alone
        if s >= max_bytes and not chunk:
            fname = os.path.join(outdir, f'chunk_{idx}.pth')
            torch.save({key: val}, fname)
            print('Wrote large single item', fname)
            # free memory
            try:
                del data[key]
            except Exception:
                pass
            idx += 1
            continue

        if chunk_size + s > max_bytes and chunk:
            fname = os.path.join(outdir, f'chunk_{idx}.pth')
            try:
                torch.save(chunk, fname)
                print('Wrote chunk', fname, 'size~', chunk_size)
                # free memory for saved keys
                for k in list(chunk.keys()):
                    try:
                        del data[k]
                    except Exception:
                        pass
            except MemoryError:
                print('MemoryError while saving chunk; falling back to per-item files')
                j = 0
                for k, v in list(chunk.items()):
                    item_fname = os.path.join(outdir, f'chunk_{idx}_item_{j}.pth')
                    torch.save({k: v}, item_fname)
                    print('Wrote item', item_fname)
                    try:
                        del data[k]
                    except Exception:
                        pass
                    j += 1
            idx += 1
            chunk = {}
            chunk_size = 0

        chunk[key] = val
        chunk_size += s

    if chunk:
        fname = os.path.join(outdir, f'chunk_{idx}.pth')
        try:
            torch.save(chunk, fname)
            print('Wrote final chunk', fname, 'size~', chunk_size)
            for k in list(chunk.keys()):
                try:
                    del data[k]
                except Exception:
                    pass
        except MemoryError:
            print('MemoryError while saving final chunk; falling back to per-item files')
            j = 0
            for k, v in list(chunk.items()):
                item_fname = os.path.join(outdir, f'chunk_{idx}_item_{j}.pth')
                torch.save({k: v}, item_fname)
                print('Wrote item', item_fname)
                try:
                    del data[k]
                except Exception:
                    pass
                j += 1


def split_list(data, outdir, max_bytes):
    chunk = []
    chunk_size = 0
    idx = 0

    # For list/tuple, iterate by index so we can delete processed items to free memory
    i = 0
    n = len(data)
    while i < n:
        item = data[i]
        s = estimate_size(item)

        if s >= max_bytes and not chunk:
            fname = os.path.join(outdir, f'chunk_{idx}.pth')
            torch.save([item], fname)
            print('Wrote large single item', fname)
            # remove item to free memory
            del data[i]
            n -= 1
            idx += 1
            continue

        if chunk_size + s > max_bytes and chunk:
            fname = os.path.join(outdir, f'chunk_{idx}.pth')
            try:
                torch.save(chunk, fname)
                print('Wrote chunk', fname, 'size~', chunk_size)
            except MemoryError:
                print('MemoryError while saving chunk; falling back to per-item files')
                j = 0
                for it in chunk:
                    item_fname = os.path.join(outdir, f'chunk_{idx}_item_{j}.pth')
                    torch.save([it], item_fname)
                    print('Wrote item', item_fname)
                    j += 1
            idx += 1
            chunk = []
            chunk_size = 0
            continue

        # Append current item to chunk and remove from original to free memory
        chunk.append(item)
        chunk_size += s
        del data[i]
        n -= 1
        # do not increment i since we deleted current index

    if chunk:
        fname = os.path.join(outdir, f'chunk_{idx}.pth')
        try:
            torch.save(chunk, fname)
            print('Wrote final chunk', fname, 'size~', chunk_size)
        except MemoryError:
            print('MemoryError while saving final chunk; falling back to per-item files')
            j = 0
            for it in chunk:
                item_fname = os.path.join(outdir, f'chunk_{idx}_item_{j}.pth')
                torch.save([it], item_fname)
                print('Wrote item', item_fname)
                j += 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--outdir', '-o', required=True)
    p.add_argument('--max-bytes', type=int, default=200*1024*1024)  # 200 MB
    args = p.parse_args()

    inp = args.input
    outdir = args.outdir
    max_bytes = args.max_bytes

    if not os.path.exists(inp):
        print('Input file not found:', inp)
        return
    ensure_dir(outdir)

    print('Loading input (this happens on local machine):', inp)
    # Some legacy .pth files require allowing certain globals (numpy scalars)
    # Use torch.serialization.safe_globals to allowlist required numpy global
    try:
        import numpy as _np
        with torch.serialization.safe_globals([_np.core.multiarray.scalar]):
            data = torch.load(inp, map_location='cpu', weights_only=False)
    except Exception as e:
        # Fallback: try loading with weights_only=False (only if file is trusted)
        print('safe_globals loading failed, retrying with weights_only=False. Error:', e)
        data = torch.load(inp, map_location='cpu', weights_only=False)
    print('Loaded. Type:', type(data))

    if isinstance(data, dict):
        split_dict(data, outdir, max_bytes)
    elif isinstance(data, (list, tuple)):
        split_list(data, outdir, max_bytes)
    else:
        # Fallback: save single objects in separate files
        print('Unknown top-level type; saving as single chunk')
        fname = os.path.join(outdir, 'chunk_0.pth')
        torch.save(data, fname)
        print('Wrote', fname)

if __name__ == '__main__':
    main()
