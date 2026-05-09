"""
Step 1c: Compute NASWOT proxy for NAS-Bench-101.

NASWOT (Mellor et al. 2021): activation diversity via covariance trace.
  - Hook ALL Conv2d layers
  - Forward pass with random input (batch=8)
  - Score = mean of covariance trace across all hooked layers

Checkpoints every CHECKPOINT_EVERY architectures.
Resumes from checkpoint on restart.

Input:  results/nasbench101/audit/arch_hashes.npy
        results/nasbench101/audit/arch_specs.pkl
Output: results/nasbench101/raw_proxy_scores/naswot.npy  shape (423624,) float64
"""

import os
import sys
import json
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from proxy_utils_101 import build_nasbench101_model, get_device

# ---------------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------------

ROOT_DIR         = Path(__file__).resolve().parents[2]
AUDIT_DIR        = ROOT_DIR / "results/nasbench101/audit"
OUT_DIR          = ROOT_DIR / "results/nasbench101/raw_proxy_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE  = OUT_DIR / "naswot_checkpoint.json"
OUT_FILE         = OUT_DIR / "naswot.npy"

CHECKPOINT_EVERY = 1000
INPUT_SIZE       = (8, 3, 32, 32)   # NASWOT: batch=8, random
C_BASE           = 16
LOG_EVERY        = 5000

# Fixed denominator: stem(1) + downsamples(2) + 9 cells × 5 nodes = 48
# Using a fixed denominator makes scores comparable across architectures
# with different numbers of Conv2d ops (variable due to maxpool3x3).
MAX_CONV_LAYERS  = 48


# ---------------------------------------------------------------------------
# NASWOT score
# ---------------------------------------------------------------------------

def naswot_score(model: nn.Module, device: torch.device) -> float:
    """
    Compute NASWOT score: mean covariance trace over all Conv2d layers.
    Returns 0.0 on any error.
    """
    try:
        model.eval()

        # Collect all Conv2d modules
        conv_modules = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if not conv_modules:
            return 0.0

        layer_activations = {i: None for i in range(len(conv_modules))}

        def make_hook(idx):
            def hook(module, inp, out):
                if isinstance(out, torch.Tensor) and out.dim() >= 2:
                    # (batch, C, H, W) -> (batch, C)
                    flat = out.view(out.size(0), out.size(1), -1).mean(dim=2)
                    layer_activations[idx] = flat.detach().cpu()
            return hook

        hooks = [m.register_forward_hook(make_hook(i))
                 for i, m in enumerate(conv_modules)]

        with torch.no_grad():
            x = torch.randn(INPUT_SIZE).to(device)
            model(x)

        for h in hooks:
            h.remove()

        layer_scores = []
        for acts in layer_activations.values():
            if acts is None or acts.size(0) < 2:
                continue
            acts = acts.float()
            acts_c = acts - acts.mean(dim=0, keepdim=True)
            cov = (acts_c.T @ acts_c) / (acts.size(0) - 1)
            layer_scores.append(cov.trace().item())

        if not layer_scores:
            return 0.0

        # Divide by MAX_CONV_LAYERS (fixed) NOT len(layer_scores) (variable).
        # This prevents architectures with fewer hooked Conv2d layers (e.g.
        # maxpool-heavy DAGs) from appearing to have higher activation diversity.
        return float(sum(layer_scores) / MAX_CONV_LAYERS)

    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = get_device()
    print(f"Using device: {device}", flush=True)
    print("Loading audit data ...", flush=True)
    arch_hashes = np.load(AUDIT_DIR / "arch_hashes.npy", allow_pickle=True)
    with open(AUDIT_DIR / "arch_specs.pkl", "rb") as f:
        arch_specs = pickle.load(f)

    n_total = len(arch_hashes)
    print(f"  Total architectures: {n_total}", flush=True)

    scores    = np.zeros(n_total, dtype=np.float64)
    start_idx = 0

    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            ckpt = json.load(f)
        start_idx = ckpt["next_idx"]
        saved = np.array(ckpt["scores"])
        scores[:len(saved)] = saved
        print(f"  Resumed from checkpoint at idx={start_idx}", flush=True)

    if start_idx >= n_total:
        print("Already complete. Saving .npy ...", flush=True)
        np.save(OUT_FILE, scores)
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()
        return

    print(f"Computing NASWOT scores [{start_idx} -> {n_total}] ...\n", flush=True)
    t0 = time.time()

    for idx in range(start_idx, n_total):
        h    = str(arch_hashes[idx])
        spec = arch_specs[h]

        model      = build_nasbench101_model(spec["adjacency"], spec["ops"], C=C_BASE).to(device)
        scores[idx] = naswot_score(model, device)
        del model

        if (idx + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t0
            rate    = (idx - start_idx + 1) / elapsed
            eta     = (n_total - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{n_total}] elapsed={elapsed/3600:.2f}h  "
                  f"ETA={eta/3600:.2f}h  rate={rate:.0f}/s", flush=True)

        if (idx + 1) % CHECKPOINT_EVERY == 0:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({"next_idx": idx + 1, "scores": scores[:idx+1].tolist()}, f)

    total_time = time.time() - t0
    print(f"\nDone in {total_time/3600:.2f}h", flush=True)

    np.save(OUT_FILE, scores)
    print(f"Saved {OUT_FILE}  shape={scores.shape}", flush=True)

    valid = scores[scores > 0]
    print(f"  Valid (>0): {len(valid)}/{n_total}", flush=True)
    if len(valid) > 0:
        print(f"  min={valid.min():.4e}  max={valid.max():.4e}  mean={valid.mean():.4e}", flush=True)

    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint deleted.", flush=True)


if __name__ == "__main__":
    main()
