"""
Step 1b: Compute SynFlow proxy for NAS-Bench-101.

SynFlow (Tanaka et al. 2020): data-free gradient-based proxy.
  1. All-ones input (deterministic, no randomness)
  2. Linearize weights: set all weights to |w|
  3. Forward pass; loss = output.sum()
  4. Backward pass
  5. Score = sum(|grad_w * w|) over all parameters
  6. Restore original weight signs

Checkpoints every CHECKPOINT_EVERY architectures.
Resumes from checkpoint on restart.

Input:  results/nasbench101/audit/arch_hashes.npy
        results/nasbench101/audit/arch_specs.pkl
Output: results/nasbench101/raw_proxy_scores/synflow.npy  shape (423624,) float64
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

ROOT_DIR         = Path("/home/anan/NAS/Experimentation/Data-Agnostic-NAS-Experimentation")
AUDIT_DIR        = ROOT_DIR / "results/nasbench101/audit"
OUT_DIR          = ROOT_DIR / "results/nasbench101/raw_proxy_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_FILE  = OUT_DIR / "synflow_checkpoint.json"
OUT_FILE         = OUT_DIR / "synflow.npy"

CHECKPOINT_EVERY = 1000
INPUT_SIZE       = (1, 3, 32, 32)   # SynFlow: batch=1, all-ones
C_BASE           = 16               # Proxy network base channels (fast)
LOG_EVERY        = 5000


# ---------------------------------------------------------------------------
# SynFlow score
# ---------------------------------------------------------------------------

def synflow_score(model: nn.Module, device: torch.device) -> float:
    """Compute SynFlow score for a model. Returns 0.0 on any error."""
    try:
        model.eval()
        model.zero_grad()

        x = torch.ones(INPUT_SIZE).to(device)

        # Linearize: store signs, set all weights to |w|
        signs = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                signs[name] = torch.sign(param.data.clone())
                param.data.abs_()

        output = model(x)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = output.sum()
        loss.backward()

        score = 0.0
        for name, param in model.named_parameters():
            if param.grad is not None:
                score += (param.grad * param.data).abs().sum().item()

        # Restore signs
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in signs:
                    param.data.mul_(signs[name])

        return float(score)

    except Exception as e:
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

    # Load checkpoint
    scores = np.zeros(n_total, dtype=np.float64)
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

    print(f"Computing SynFlow scores [{start_idx} -> {n_total}] ...\n", flush=True)
    t0 = time.time()

    for idx in range(start_idx, n_total):
        h = str(arch_hashes[idx])
        spec = arch_specs[h]

        model = build_nasbench101_model(spec["adjacency"], spec["ops"], C=C_BASE).to(device)
        scores[idx] = synflow_score(model, device)
        del model

        # Log
        if (idx + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t0
            rate    = (idx - start_idx + 1) / elapsed
            eta     = (n_total - idx - 1) / rate if rate > 0 else 0
            print(f"  [{idx+1}/{n_total}] elapsed={elapsed/3600:.2f}h  "
                  f"ETA={eta/3600:.2f}h  rate={rate:.0f}/s", flush=True)

        # Checkpoint
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            with open(CHECKPOINT_FILE, "w") as f:
                json.dump({"next_idx": idx + 1, "scores": scores[:idx+1].tolist()}, f)

    total_time = time.time() - t0
    print(f"\nDone in {total_time/3600:.2f}h", flush=True)

    # Save final output
    np.save(OUT_FILE, scores)
    print(f"Saved {OUT_FILE}  shape={scores.shape}", flush=True)

    # Report stats (skip zeros)
    valid = scores[scores > 0]
    print(f"  Valid (>0): {len(valid)}/{n_total}", flush=True)
    print(f"  min={valid.min():.4e}  max={valid.max():.4e}  mean={valid.mean():.4e}", flush=True)

    # Delete checkpoint
    if CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint deleted.", flush=True)


if __name__ == "__main__":
    main()
