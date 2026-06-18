"""
Proxy 3 (SSS): Zen-Score
Measures activation diversity via covariance trace under multiple random inputs.

Direction check (Step 0 audit): expected positive ρ vs GT accuracy (same direction as NASWOT).
→ NO negation applied here.
→ log(score + ε) transform applied in transform_proxies_sss.py (Step 3).

Implementation: multi-layer, multi-sample hooking
  - 4 forward passes with fresh random batches (batch=4 each)
  - Hook ALL Conv2d outputs → compute covariance trace per layer per pass
  - Final score = mean of all (layer × sample) traces

Output: results/nats_bench_sss/raw_proxy_scores/zenscore.npy
  Shape (32768,) — raw (un-negated, un-logged) Zen-Score per arch index.
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

NUM_SAMPLES = 4
BATCH_SIZE  = 4
INPUT_SIZE  = (BATCH_SIZE, 3, 32, 32)


def zenscore(model: nn.Module) -> float:
    """Mean covariance trace across all Conv2d layers and multiple random inputs."""
    model.eval()

    conv_mods = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not conv_mods:
        return 0.0

    all_scores = []

    with torch.no_grad():
        for _ in range(NUM_SAMPLES):
            activations = {}

            def make_hook(layer_idx):
                def hook(module, inp, out):
                    if isinstance(out, torch.Tensor):
                        flat = out.detach().view(out.size(0), out.size(1), -1).mean(dim=2)
                        activations[layer_idx] = flat.cpu().float()
                return hook

            hooks = [m.register_forward_hook(make_hook(i)) for i, m in enumerate(conv_mods)]
            x = torch.randn(INPUT_SIZE)
            model(x)
            for h in hooks:
                h.remove()

            for i in range(len(conv_mods)):
                acts = activations.get(i)
                if acts is None or acts.size(0) < 2:
                    continue
                acts_c = acts - acts.mean(dim=0, keepdim=True)
                cov = (acts_c.t() @ acts_c) / acts.size(0)
                trace = torch.diagonal(cov).sum().item()
                if trace > 0.0:
                    all_scores.append(trace)

    return float(np.mean(all_scores)) if all_scores else 0.0


def compute_all(verbose: bool = True) -> np.ndarray:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    scores = np.zeros(N_ARCHS, dtype=np.float64)

    for idx in range(N_ARCHS):
        pkl_path = os.path.join(ARCH_DIR, "%d.pickle" % idx)
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        arch_str = data["90"]["arch_str"]

        model = build_sss_model(arch_str)
        scores[idx] = zenscore(model)
        del model

        if verbose and (idx + 1) % 500 == 0:
            print("  [zenscore] %d / %d  (last arch=%s, score=%.4f)"
                  % (idx + 1, N_ARCHS, arch_str, scores[idx]), flush=True)

    out_path = os.path.join(OUTPUT_DIR, "zenscore.npy")
    np.save(out_path, scores)
    print("\nSaved: %s" % out_path, flush=True)
    print("  min=%.4f  max=%.4f  mean=%.4f  zeros=%d"
          % (scores.min(), scores.max(), scores.mean(), (scores == 0).sum()), flush=True)
    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Zen-Score proxy for all SSS arches")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    compute_all(verbose=not args.quiet)
