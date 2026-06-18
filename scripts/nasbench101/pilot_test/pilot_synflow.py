"""
Pilot: SynFlow proxy on first N_PILOT=20,000 architectures.
Estimated time: ~15 min.
Output: results/nasbench101/pilot_test/raw_proxy_scores/synflow.npy  shape (20000,)
"""
import sys, time, pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from proxy_utils_101 import build_nasbench101_model

N_PILOT   = 20_000
C_BASE    = 16
IN_SIZE   = (1, 3, 32, 32)
LOG_EVERY = 1000

ROOT_DIR  = Path("F:/Thesis/Experimentation")
AUDIT_DIR = ROOT_DIR / "results/nasbench101/audit"
OUT_DIR   = ROOT_DIR / "results/nasbench101/pilot_test/raw_proxy_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def synflow_score(model: nn.Module) -> float:
    try:
        model.eval(); model.zero_grad()
        x = torch.ones(IN_SIZE)
        signs = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                signs[name] = torch.sign(p.data.clone())
                p.data.abs_()
        out = model(x)
        if isinstance(out, (tuple, list)): out = out[0]
        out.sum().backward()
        score = sum((p.grad * p.data).abs().sum().item()
                    for p in model.parameters() if p.grad is not None)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if name in signs: p.data.mul_(signs[name])
        return float(score)
    except Exception:
        return 0.0


def main():
    arch_hashes = np.load(AUDIT_DIR / "arch_hashes.npy", allow_pickle=True)[:N_PILOT]
    with open(AUDIT_DIR / "arch_specs.pkl", "rb") as f:
        arch_specs = pickle.load(f)

    scores = np.zeros(N_PILOT, dtype=np.float64)
    t0 = time.time()
    print(f"Computing SynFlow on {N_PILOT} architectures ...", flush=True)

    for idx, h in enumerate(arch_hashes):
        spec = arch_specs[str(h)]
        model = build_nasbench101_model(spec["adjacency"], spec["ops"], C=C_BASE)
        scores[idx] = synflow_score(model)
        del model
        if (idx + 1) % LOG_EVERY == 0:
            elapsed = time.time() - t0
            eta = (N_PILOT - idx - 1) / ((idx + 1) / elapsed)
            print(f"  [{idx+1}/{N_PILOT}] {elapsed:.0f}s elapsed  ETA {eta:.0f}s", flush=True)

    np.save(OUT_DIR / "synflow.npy", scores)
    print(f"Done in {time.time()-t0:.0f}s.  Saved synflow.npy  "
          f"valid={int((scores>0).sum())}/{N_PILOT}", flush=True)


if __name__ == "__main__":
    main()
