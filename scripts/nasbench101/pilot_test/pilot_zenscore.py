"""
Pilot: ZenScore proxy on first N_PILOT=20,000 architectures.
Estimated time: ~18 min.
Output: results/nasbench101/pilot_test/raw_proxy_scores/zenscore.npy  shape (20000,)
"""
import sys, time, pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from proxy_utils_101 import build_nasbench101_model

N_PILOT     = 20_000
C_BASE      = 16
IN_SIZE     = (4, 3, 32, 32)
NUM_SAMPLES = 4
LOG_EVERY   = 1000

ROOT_DIR  = Path("F:/Thesis/Experimentation")
AUDIT_DIR = ROOT_DIR / "results/nasbench101/audit"
OUT_DIR   = ROOT_DIR / "results/nasbench101/pilot_test/raw_proxy_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def zenscore(model: nn.Module) -> float:
    try:
        model.eval()
        conv_modules = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
        if not conv_modules:
            return 0.0
        all_scores = []
        with torch.no_grad():
            for _ in range(NUM_SAMPLES):
                acts_per_layer = {i: None for i in range(len(conv_modules))}

                def make_hook(idx):
                    def hook(m, inp, out):
                        if isinstance(out, torch.Tensor) and out.dim() >= 2:
                            flat = out.view(out.size(0), out.size(1), -1).mean(dim=2)
                            acts_per_layer[idx] = flat.detach().cpu()
                    return hook

                hooks = [m.register_forward_hook(make_hook(i))
                         for i, m in enumerate(conv_modules)]
                model(torch.randn(IN_SIZE))
                for h in hooks: h.remove()

                for acts in acts_per_layer.values():
                    if acts is None or acts.size(0) < 2: continue
                    acts = acts.float()
                    a = acts - acts.mean(0, keepdim=True)
                    cov = (a.T @ a) / (acts.size(0) - 1)
                    all_scores.append(cov.trace().item())
        return float(np.mean(all_scores)) if all_scores else 0.0
    except Exception:
        return 0.0


def main():
    arch_hashes = np.load(AUDIT_DIR / "arch_hashes.npy", allow_pickle=True)[:N_PILOT]
    with open(AUDIT_DIR / "arch_specs.pkl", "rb") as f:
        arch_specs = pickle.load(f)

    scores = np.zeros(N_PILOT, dtype=np.float64)
    t0 = time.time()
    print(f"Computing ZenScore on {N_PILOT} architectures ...", flush=True)

    for idx, h in enumerate(arch_hashes):
        spec = arch_specs[str(h)]
        model = build_nasbench101_model(spec["adjacency"], spec["ops"], C=C_BASE)
        scores[idx] = zenscore(model)
        del model
        if (idx + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t0
            eta = (N_PILOT - idx - 1) / ((idx + 1) / elapsed)
            print(f"  [{idx+1}/{N_PILOT}] {elapsed:.0f}s elapsed  ETA {eta:.0f}s", flush=True)

    np.save(OUT_DIR / "zenscore.npy", scores)
    print(f"Done in {time.time()-t0:.0f}s.  Saved zenscore.npy  "
          f"valid={int((scores>0).sum())}/{N_PILOT}", flush=True)


if __name__ == "__main__":
    main()
