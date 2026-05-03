"""
Pilot test orchestrator for NAS-Bench-101.
Runs the full pipeline sequentially on N=20,000 architectures.

Estimated total time: ~60-70 min (proxies dominate)
  - SynFlow:   ~15 min
  - NASWOT:    ~15 min
  - ZenScore:  ~18 min
  - Pipeline:  ~2 min
  - MLP:       ~3 min

Usage:
  python scripts/nasbench101/pilot_test/run_all_pilot.py

To skip re-computing proxies (if they already exist):
  python scripts/nasbench101/pilot_test/run_all_pilot.py --skip-proxies
"""

import sys
import time
import subprocess
from pathlib import Path

ROOT_DIR  = Path("F:/Thesis/Experimentation")
PILOT_DIR = ROOT_DIR / "results/nasbench101/pilot_test/raw_proxy_scores"
PYTHON    = str(ROOT_DIR / "envs/nasbench_env/Scripts/python.exe")
SCRIPTS   = ROOT_DIR / "scripts/nasbench101/pilot_test"

skip_proxies = "--skip-proxies" in sys.argv


def run(script: str, label: str):
    print(f"\n{'='*60}", flush=True)
    print(f"Running: {label}", flush=True)
    print(f"{'='*60}", flush=True)
    t0 = time.time()
    result = subprocess.run([PYTHON, str(SCRIPTS / script)], check=False)
    elapsed = time.time() - t0
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n[{label}] {status} in {elapsed/60:.1f} min", flush=True)
    if result.returncode != 0:
        print("Stopping pilot run due to failure.", flush=True)
        sys.exit(1)
    return elapsed


t_start = time.time()

if not skip_proxies:
    # Proxy scripts (sequential -- if you want parallel, run them manually in 3 terminals)
    run("pilot_synflow.py",  "SynFlow (N=20k)")
    run("pilot_naswot.py",   "NASWOT  (N=20k)")
    run("pilot_zenscore.py", "ZenScore (N=20k)")
else:
    print("Skipping proxy computation (--skip-proxies flag set).", flush=True)
    for name in ["synflow", "naswot", "zenscore"]:
        p = PILOT_DIR / f"{name}.npy"
        if not p.exists():
            print(f"ERROR: {p} not found.  Run without --skip-proxies first.")
            sys.exit(1)

run("pilot_pipeline.py", "Steps 2-6 (transform -> PCA)")
run("pilot_mlp.py",      "Step 7 (MLP ablation)")

total = time.time() - t_start
print(f"\n{'='*60}", flush=True)
print(f"PILOT COMPLETE in {total/60:.1f} min", flush=True)
print(f"Results in: results/nasbench101/pilot_test/", flush=True)
print(f"  Key output: pilot_test/surrogate_mlp/ablation_table.json", flush=True)
print(f"{'='*60}", flush=True)
