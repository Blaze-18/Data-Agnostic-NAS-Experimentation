r"""
Decompress NATS-sss-v1_0-50262.pickle.pbz2
===========================================

Reads the bzip2-compressed pickle file and extracts each architecture's data
as an individual {index}.pickle file, matching the NATS-Bench "simple archive"
format required for fast_mode=True API access.

File structure (confirmed by inspection):
  top-level dict with 4 keys:
    'meta_archs'        : list[32768]  — arch strings e.g. '8:8:8:8:8'
    'total_archs'       : int = 32768
    'arch2infos'        : dict{0..32767} -> dict{'01':..., '12':..., '90':...}
    'evaluated_indexes' : set of all 32768 indices

  Each arch2infos[idx] has keys '01', '12', '90' (epoch counts), each
  containing: arch_index, arch_str, all_results, dataset_seed, clear_net_done.

Expected input  : data/nats_bench_sss/NATS-sss-v1_0-50262.pickle.pbz2
Expected output : data/nats_bench_sss/NATS-sss-v1_0-50262-simple/{index}.pickle
                  (32,768 files, one per architecture)

Timing note:
  The bzip2 decompression + pickle load is the slow step (~5-20 min depending
  on disk speed and CPU). A heartbeat thread prints every 30 s so you can
  verify the process is alive.

Usage:
  python scripts/nats_bench_sss/decompress_nats_sss.py

  Optional overrides:
    --input     path to .pickle.pbz2 file
    --output    path to output directory
    --no-verify skip the file-count verification phase
"""

import argparse
import bz2
import os
import pickle
import sys
import threading
import time
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_INPUT  = Path("data/nats_bench_sss/NATS-sss-v1_0-50262.pickle.pbz2")
DEFAULT_OUTPUT = Path("data/nats_bench_sss/NATS-sss-v1_0-50262-simple")
EXPECTED_COUNT = 32_768   # SSS search space size


# ─────────────────────────────────────────────────────────────────────────────
# HEARTBEAT THREAD
# ─────────────────────────────────────────────────────────────────────────────

class Heartbeat:
    """
    Prints a timestamped "still working" message every `interval` seconds
    until stopped. Use as a context manager.
    """
    def __init__(self, message: str = "Still working...", interval: int = 30):
        self._message  = message
        self._interval = interval
        self._stop     = threading.Event()
        self._thread   = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self._interval):
            elapsed = time.strftime("%H:%M:%S")
            print(f"  [{elapsed}] {self._message}", flush=True)

    def __enter__(self):
        self._start = time.monotonic()
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=1)
        elapsed = time.monotonic() - self._start
        print(f"  Done in {elapsed:.1f} s", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — LOAD
# ─────────────────────────────────────────────────────────────────────────────

def load_compressed(input_path: Path) -> dict:
    """
    Decompress the .pbz2 file, unpickle it, and return the arch2infos dict.

    The top-level pickle is a 4-key dict; we extract only 'arch2infos'
    (the 32,768-entry dict keyed by integer arch index) and let the rest
    be garbage-collected to free RAM before extraction begins.

    Returns
    -------
    arch2infos : dict
        Keys are integer architecture indices (0 ... 32767).
        Values are dicts with keys '01', '12', '90' (epoch variants).
    """
    size_mb = input_path.stat().st_size / (1024 ** 2)
    print(f"\n{'=' * 70}")
    print(f"PHASE 1 — LOAD & DECOMPRESS")
    print(f"{'=' * 70}")
    print(f"  Input file : {input_path}")
    print(f"  Size       : {size_mb:.1f} MB (compressed)")
    print()
    print("  Opening bzip2 stream and unpickling — this is the slow step.")
    print("  A heartbeat is printed every 30 s so you can confirm progress.")
    print("  Do NOT interrupt during this phase.\n")

    with Heartbeat("bzip2 decompression + pickle load in progress..."):
        with bz2.BZ2File(input_path, "rb") as f:
            raw = pickle.load(f)

    print()
    if not isinstance(raw, dict):
        raise TypeError(
            f"Expected a dict from the pickle, got {type(raw)}. "
            f"The file may be corrupt or a different format."
        )

    if "arch2infos" not in raw:
        raise KeyError(
            f"Key 'arch2infos' not found. Top-level keys: {list(raw.keys())}"
        )

    arch2infos = raw["arch2infos"]
    n = len(arch2infos)
    print(f"  Top-level keys  : {list(raw.keys())}")
    print(f"  arch2infos size : {n:,} entries")
    if n != EXPECTED_COUNT:
        print(
            f"  WARNING: expected {EXPECTED_COUNT:,} entries for SSS, "
            f"got {n:,}. Continuing anyway."
        )

    # Show structure of one entry
    sample_idx = next(iter(arch2infos))
    sample_val = arch2infos[sample_idx]
    print(f"  Sample index    : {sample_idx}")
    print(f"  Sample keys     : {list(sample_val.keys())}")

    # Free the rest of the top-level dict
    del raw

    return arch2infos


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — EXTRACT
# ─────────────────────────────────────────────────────────────────────────────

def extract_to_simple_archive(arch2infos: dict, output_dir: Path) -> None:
    """
    Write each architecture entry as {index}.pickle into output_dir.

    This matches the NATS-Bench "simple archive" format used by
    fast_mode=True. Files are plain (uncompressed) pickles for fast
    random access during proxy computation.
    """
    print(f"\n{'=' * 70}")
    print(f"PHASE 2 — EXTRACT TO INDIVIDUAL FILES")
    print(f"{'=' * 70}")
    print(f"  Output dir : {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    n          = len(arch2infos)
    report_at  = max(1, n // 20)   # print progress every 5%
    t_start    = time.monotonic()

    print(f"  Writing {n:,} .pickle files ...\n")

    for written, (idx, arch_data) in enumerate(sorted(arch2infos.items()), start=1):
        out_path = output_dir / f"{idx}.pickle"
        with open(out_path, "wb") as f:
            pickle.dump(arch_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        if written % report_at == 0 or written == n:
            elapsed   = time.monotonic() - t_start
            pct       = 100.0 * written / n
            rate      = written / elapsed if elapsed > 0 else 0
            remaining = (n - written) / rate if rate > 0 else 0
            print(
                f"  [{time.strftime('%H:%M:%S')}]  {written:>6,}/{n:,}  "
                f"({pct:5.1f}%)  "
                f"{rate:6.0f} files/s  "
                f"ETA {remaining:5.0f} s",
                flush=True,
            )

    elapsed = time.monotonic() - t_start
    print(f"\n  Extraction complete in {elapsed:.1f} s.")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — VERIFY
# ─────────────────────────────────────────────────────────────────────────────

def verify(output_dir: Path) -> None:
    print(f"\n{'=' * 70}")
    print(f"PHASE 3 — VERIFY")
    print(f"{'=' * 70}")

    files = list(output_dir.glob("*.pickle"))
    n     = len(files)
    print(f"  Files found : {n:,}")

    if n == EXPECTED_COUNT:
        print(f"  Count check : PASS  ({n:,} == {EXPECTED_COUNT:,})")
    else:
        print(
            f"  Count check : FAIL  ({n:,} != {EXPECTED_COUNT:,})\n"
            f"  Missing indices may indicate a corrupt source file or "
            f"interrupted extraction."
        )

    # Spot-check: try loading a few files
    spot_indices = [0, 1000, 16383, 32767]
    all_ok = True
    print()
    for idx in spot_indices:
        fpath = output_dir / f"{idx}.pickle"
        if fpath.exists():
            try:
                with open(fpath, "rb") as f:
                    obj = pickle.load(f)
                status = f"OK  ({type(obj).__name__})"
            except Exception as e:
                status = f"LOAD ERROR: {e}"
                all_ok = False
        else:
            status = "MISSING"
            all_ok = False
        print(f"  index {idx:>5} : {status}")

    print()
    if all_ok and n == EXPECTED_COUNT:
        print("  Overall: PASS — archive is complete and readable.")
    else:
        print("  Overall: FAIL — see issues above.")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Decompress NATS-sss .pickle.pbz2 to simple archive format."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to .pickle.pbz2 file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for .pickle files (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the verification phase.",
    )
    args = parser.parse_args()

    print("\n" + "#" * 70)
    print("# NATS-Bench SSS — DECOMPRESSION")
    print(f"#   Input  : {args.input}")
    print(f"#   Output : {args.output}")
    print("#" * 70)

    # --- Pre-flight check ---
    if not args.input.exists():
        print(f"\nERROR: Input file not found: {args.input}")
        sys.exit(1)

    t_total = time.monotonic()

    arch2infos = load_compressed(args.input)
    extract_to_simple_archive(arch2infos, args.output)

    if not args.no_verify:
        verify(args.output)

    total = time.monotonic() - t_total
    print(f"\n{'#' * 70}")
    print(f"# COMPLETE  —  total time {total:.1f} s  ({total/60:.1f} min)")
    print(f"# Simple archive: {args.output}")
    print("#" * 70)


if __name__ == "__main__":
    main()
