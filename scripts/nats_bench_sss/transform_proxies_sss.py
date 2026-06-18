"""
Step 2 -- Log-transform raw proxy scores (NATS-Bench SSS).

Transforms:
  param_count : log(x)            -- always positive, no offset needed
  naswot      : log(x + 1e-10)   -- scores ~1e-4 to ~2e-3, tiny offset for safety
  zenscore    : log(x + 1e-10)   -- same range as naswot
  synflow     : log(x)            -- scores ~1e10 to ~5e17, all positive

No negation applied: all proxies have confirmed positive Spearman rho vs GT.
"""

import numpy as np
import os, json, sys

RAW_DIR   = "results/nats_bench_sss/raw_proxy_scores"
OUT_DIR   = "results/nats_bench_sss/transformed_proxy"
AUDIT_DIR = "results/nats_bench_sss/audit"
EPS       = 1e-10

PROXIES = {
    "param_count": {"file": "param_count.npy", "transform": "log",        "eps": 0.0},
    "naswot"     : {"file": "naswot.npy",       "transform": "log_offset", "eps": EPS},
    "zenscore"   : {"file": "zenscore.npy",     "transform": "log_offset", "eps": EPS},
    "synflow"    : {"file": "synflow.npy",       "transform": "log",        "eps": 0.0},
}

os.makedirs(OUT_DIR,   exist_ok=True)
os.makedirs(AUDIT_DIR, exist_ok=True)


def transform(name, arr, cfg):
    eps = cfg["eps"]
    if cfg["transform"] == "log":
        assert (arr > 0).all(), f"{name}: non-positive values before log"
        return np.log(arr)
    else:  # log_offset
        return np.log(arr + eps)


def main():
    print("=" * 60)
    print("Step 2 -- Transform proxy scores (NATS-Bench SSS)")
    print("=" * 60)

    stats = {}
    for name, cfg in PROXIES.items():
        raw_path = os.path.join(RAW_DIR, cfg["file"])
        raw = np.load(raw_path)
        print(f"\n  {name}")
        print(f"    raw  : min={raw.min():.4e}  max={raw.max():.4e}  mean={raw.mean():.4e}")

        t = transform(name, raw, cfg)
        print(f"    log  : min={t.min():.4f}   max={t.max():.4f}   mean={t.mean():.4f}  std={t.std():.4f}")

        out_path = os.path.join(OUT_DIR, f"{name}.npy")
        np.save(out_path, t)
        print(f"    saved -> {out_path}")

        stats[name] = {
            "raw_min": float(raw.min()), "raw_max": float(raw.max()),
            "log_min": float(t.min()),   "log_max": float(t.max()),
            "log_mean": float(t.mean()), "log_std": float(t.std()),
        }

    # Save audit JSON
    audit = {"step": "transform", "proxies": stats}
    audit_path = os.path.join(AUDIT_DIR, "transform_stats.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)
    print(f"\n  Audit -> {audit_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
