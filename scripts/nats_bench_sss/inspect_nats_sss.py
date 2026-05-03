"""
Inspect the top-level structure of NATS-sss-v1_0-50262.pickle.pbz2
without trying to save anything.

Run this first to understand the key names and nested layout so the
decompress script can be fixed accordingly.

Usage:
    envs\nasbench_env\Scripts\python.exe scripts\nats_bench_sss\inspect_nats_sss.py
"""

import bz2
import pickle
import threading
import time
from pathlib import Path

INPUT = Path("data/nats_bench_sss/NATS-sss-v1_0-50262.pickle.pbz2")


class Heartbeat:
    def __init__(self, interval=30):
        self._stop   = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._interval = interval

    def _run(self):
        while not self._stop.wait(self._interval):
            print(f"  [{time.strftime('%H:%M:%S')}] Still loading...", flush=True)

    def __enter__(self):
        self._t0 = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=1)
        print(f"  Done in {time.monotonic() - self._t0:.1f} s", flush=True)


def inspect(obj, prefix="", depth=0, max_depth=3, max_items=3):
    """Recursively print the structure of obj."""
    indent = "  " * depth
    t = type(obj).__name__

    if depth > max_depth:
        print(f"{indent}{prefix}<truncated>")
        return

    if isinstance(obj, dict):
        print(f"{indent}{prefix}dict  ({len(obj)} keys)")
        for i, (k, v) in enumerate(obj.items()):
            if i >= max_items:
                print(f"{indent}  ... ({len(obj) - max_items} more keys)")
                break
            inspect(v, prefix=f"[{k!r}] -> ", depth=depth+1,
                    max_depth=max_depth, max_items=max_items)

    elif isinstance(obj, (list, tuple)):
        tname = type(obj).__name__
        n = len(obj)
        print(f"{indent}{prefix}{tname}  (len={n})")
        if n > 0:
            inspect(obj[0], prefix="[0] -> ", depth=depth+1,
                    max_depth=max_depth, max_items=max_items)
            if n > 1:
                inspect(obj[-1], prefix=f"[{n-1}] -> ", depth=depth+1,
                        max_depth=max_depth, max_items=max_items)

    elif hasattr(obj, '__dict__'):
        d = obj.__dict__
        print(f"{indent}{prefix}{t}  (attrs: {list(d.keys())[:6]})")

    else:
        # Scalar / leaf
        val_repr = repr(obj)
        if len(val_repr) > 80:
            val_repr = val_repr[:80] + "..."
        print(f"{indent}{prefix}{t}  = {val_repr}")


def main():
    print(f"\nLoading {INPUT} ...")
    with Heartbeat():
        with bz2.BZ2File(INPUT, "rb") as f:
            data = pickle.load(f)

    print("\n" + "=" * 70)
    print("TOP-LEVEL STRUCTURE")
    print("=" * 70)
    inspect(data, max_depth=4, max_items=5)

    # Extra: find any list that has exactly 32768 entries
    print("\n" + "=" * 70)
    print("SEARCHING FOR 32768-LENGTH LISTS AT DEPTH 1")
    print("=" * 70)
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (list, tuple)) and len(v) == 32768:
                print(f"  KEY '{k}' has len=32768  -> this is the arch list")
                # Show element 0 deeply
                print(f"\n  data['{k}'][0] structure:")
                inspect(v[0], prefix="  ", depth=1, max_depth=4, max_items=4)
            else:
                n = len(v) if hasattr(v, "__len__") else "N/A"
                print(f"  KEY '{k}'  type={type(v).__name__}  len={n}")


if __name__ == "__main__":
    main()
