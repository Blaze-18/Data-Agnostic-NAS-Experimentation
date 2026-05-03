"""
Proxy 4 (SSS): SynFlow
Data-free pruning-at-initialization score (Tanaka et al., 2020).

NOTE: Step 0 audit (NAS-Bench-201 analogy) predicts SynFlow will be EXCLUDED
at Step 5 (partial rank Spearman ≈ 0 after param_count debiasing).  This script
is provided for completeness and ablation study only.

Algorithm (original Tanaka et al.):
  1. All-ones input (data-free, deterministic)
  2. Linearize network: store weight signs, set all weights to |w|
     → prevents gradient cancellation from mixed-sign weights
  3. Forward pass → loss = output.sum()
  4. Backward pass
  5. Score = Σ |grad_w × w| over all parameters
  6. Restore original weight signs

Output: results/nats_bench_sss/raw_proxy_scores/synflow.npy
  Shape (32768,) — SynFlow score per arch index.
"""
import os
import sys
import pickle
import warnings
import argparse
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from proxy_utils_sss import build_sss_model

# ─── Paths ───────────────────────────────────────────────────────────────────
ARCH_DIR    = "data/nats_bench_sss/NATS-sss-v1_0-50262-simple"
OUTPUT_DIR  = "results/nats_bench_sss/raw_proxy_scores"
N_ARCHS     = 32768

INPUT_SIZE  = (1, 3, 32, 32)


def synflow_score(model: nn.Module) -> float:
    """
    Compute SynFlow score using the original data-free algorithm.
    Uses all-ones input and weight linearization to prevent gradient cancellation.
    """
    model.eval()
    model.zero_grad()

    # Step 1: All-ones input
    x = torch.ones(INPUT_SIZE)

    # Step 2: Linearize — store signs, make all weights positive
    signs = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            signs[name] = torch.sign(param.data.clone())
            param.data.abs_()

    # Step 3–4: Forward + backward
    output = model(x)
    if isinstance(output, (tuple, list)):
        output = output[0]
    loss = output.sum()
    loss.backward()

    # Step 5: Score = Σ |grad × weight|
    score = 0.0
    for name, param in model.named_parameters():
        if param.grad is not None:
            score += (param.grad * param.data).abs().sum().item()

    # Step 6: Restore original signs
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in signs:
                param.data.mul_(signs[name])

    model.zero_grad()
    return float(score)


def compute_all(verbose: bool = True) -> np.ndarray:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scores = np.zeros(N_ARCHS, dtype=np.float64)

    for idx in range(N_ARCHS):
        pkl_path = os.path.join(ARCH_DIR, "%d.pickle" % idx)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        arch_str = data["90"]["arch_str"]

        model = build_sss_model(arch_str)
        scores[idx] = synflow_score(model)
        del model

        if verbose and (idx + 1) % 500 == 0:
            print("  [synflow] %d / %d  (last arch=%s, score=%.4f)"
                  % (idx + 1, N_ARCHS, arch_str, scores[idx]), flush=True)

    out_path = os.path.join(OUTPUT_DIR, "synflow.npy")
    np.save(out_path, scores)
    print("\nSaved: %s" % out_path, flush=True)
    print("  min=%.4f  max=%.4f  mean=%.4f  zeros=%d"
          % (scores.min(), scores.max(), scores.mean(), (scores == 0).sum()), flush=True)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute SynFlow proxy for all SSS arches")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    compute_all(verbose=not args.quiet)
