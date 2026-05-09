"""
Step 2: Log-transform raw proxy scores for NAS-Bench-101.

Transformations:
  param_count  -> log(x)
  synflow      -> log(x + 1e-8)   [handles near-zero values]
  naswot       -> sign-check first: if raw Spearman < 0, negate then log(-x + eps)
  zenscore     -> same sign-check as naswot

Sign check uses raw Spearman rho(proxy, GT). If negative, negate before log so
that transformed proxy is positively correlated with GT.

Output: results/nasbench101/transformed_proxy/
  param_count_log.npy, synflow_log.npy, naswot_log.npy, zenscore_log.npy
  transform_summary.json
"""

import numpy as np
import json
from pathlib import Path
from scipy.stats import spearmanr

ROOT_DIR   = Path("/home/anan/NAS/Experimentation/Data-Agnostic-NAS-Experimentation")
AUDIT_DIR  = ROOT_DIR / "results/nasbench101/audit"
RAW_DIR    = ROOT_DIR / "results/nasbench101/raw_proxy_scores"
OUT_DIR    = ROOT_DIR / "results/nasbench101/transformed_proxy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EPS = 1e-8


def log_transform(x: np.ndarray, negate: bool = False, eps: float = EPS) -> np.ndarray:
    """
    Apply log(x + eps), then optionally negate the result.
    Handles Inf/NaN by replacing with max finite value before transform.
    
    If negate=True: returns -log(x + eps)
    If negate=False: returns log(x + eps)
    """
    x = x.copy()
    
    # Handle Inf/NaN values
    finite_mask = np.isfinite(x)
    if not finite_mask.all():
        max_finite = x[finite_mask].max() if finite_mask.any() else 1e10
        x[~finite_mask] = max_finite
        print(f"    Replaced {(~finite_mask).sum()} Inf/NaN values with {max_finite:.4e}")
    
    # Clamp to minimum epsilon
    x = np.where(x < eps, eps, x)
    
    # Take log
    result = np.log(x)
    
    # Negate AFTER log if needed (to flip correlation direction)
    if negate:
        result = -result
    
    return result


def main():
    gt = np.load(AUDIT_DIR / "gt_accuracies.npy").astype(np.float64)
    print(f"GT loaded: shape={gt.shape}", flush=True)

    summary = {}

    # ------------------------------------------------------------------
    # param_count
    # ------------------------------------------------------------------
    raw_pc = np.load(RAW_DIR / "param_count.npy")
    t_pc   = log_transform(raw_pc)
    np.save(OUT_DIR / "param_count_log.npy", t_pc.astype(np.float32))
    rho_pc, _ = spearmanr(t_pc, gt)
    print(f"param_count: raw_min={raw_pc.min():.0f}  log_rho={rho_pc:.4f}", flush=True)
    summary["param_count"] = {"raw_rho": float(rho_pc), "negate": False, "transform": "log"}

    # ------------------------------------------------------------------
    # synflow
    # ------------------------------------------------------------------
    raw_sf = np.load(RAW_DIR / "synflow.npy")
    rho_sf_raw, _ = spearmanr(raw_sf, gt)
    negate_sf = bool(rho_sf_raw < 0)
    t_sf = log_transform(raw_sf, negate=negate_sf)
    np.save(OUT_DIR / "synflow_log.npy", t_sf.astype(np.float32))
    rho_sf, _ = spearmanr(t_sf, gt)
    print(f"synflow:     raw_rho={rho_sf_raw:.4f}  negate={negate_sf}  "
          f"log_rho={rho_sf:.4f}", flush=True)
    summary["synflow"] = {"raw_rho": float(rho_sf_raw), "negate": negate_sf,
                          "transform": "negate+log" if negate_sf else "log",
                          "transformed_rho": float(rho_sf)}

    # ------------------------------------------------------------------
    # naswot
    # ------------------------------------------------------------------
    raw_nw = np.load(RAW_DIR / "naswot.npy")
    rho_nw_raw, _ = spearmanr(raw_nw, gt)
    negate_nw = bool(rho_nw_raw < 0)
    t_nw = log_transform(raw_nw, negate=negate_nw)
    np.save(OUT_DIR / "naswot_log.npy", t_nw.astype(np.float32))
    rho_nw, _ = spearmanr(t_nw, gt)
    print(f"naswot:      raw_rho={rho_nw_raw:.4f}  negate={negate_nw}  "
          f"log_rho={rho_nw:.4f}", flush=True)
    summary["naswot"] = {"raw_rho": float(rho_nw_raw), "negate": negate_nw,
                         "transform": "negate+log" if negate_nw else "log",
                         "transformed_rho": float(rho_nw)}

    # ------------------------------------------------------------------
    # zenscore
    # ------------------------------------------------------------------
    raw_zs = np.load(RAW_DIR / "zenscore.npy")
    rho_zs_raw, _ = spearmanr(raw_zs, gt)
    negate_zs = bool(rho_zs_raw < 0)
    t_zs = log_transform(raw_zs, negate=negate_zs)
    np.save(OUT_DIR / "zenscore_log.npy", t_zs.astype(np.float32))
    rho_zs, _ = spearmanr(t_zs, gt)
    print(f"zenscore:    raw_rho={rho_zs_raw:.4f}  negate={negate_zs}  "
          f"log_rho={rho_zs:.4f}", flush=True)
    summary["zenscore"] = {"raw_rho": float(rho_zs_raw), "negate": negate_zs,
                           "transform": "negate+log" if negate_zs else "log",
                           "transformed_rho": float(rho_zs)}

    with open(OUT_DIR / "transform_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved transform_summary.json to {OUT_DIR}", flush=True)
    
    # ------------------------------------------------------------------
    # Post-transformation validation
    # ------------------------------------------------------------------
    print("\n" + "="*60)
    print("POST-TRANSFORMATION VALIDATION")
    print("="*60)
    
    all_valid = True
    
    for proxy_name in ["param_count", "synflow", "naswot", "zenscore"]:
        t_data = np.load(OUT_DIR / f"{proxy_name}_log.npy")
        
        has_nan = np.isnan(t_data).sum()
        has_inf = np.isinf(t_data).sum()
        
        print(f"\n{proxy_name.upper()}:")
        print(f"  Shape: {t_data.shape}")
        print(f"  Dtype: {t_data.dtype}")
        print(f"  Range: [{t_data.min():.4f}, {t_data.max():.4f}]")
        print(f"  Mean: {t_data.mean():.4f}, Std: {t_data.std():.4f}")
        print(f"  NaN: {has_nan}, Inf: {has_inf}")
        
        if has_nan > 0 or has_inf > 0:
            print(f"  ⚠️  WARNING: Contains {has_nan} NaN and {has_inf} Inf values!")
            all_valid = False
        else:
            print(f"  ✅ Clean (no NaN/Inf)")
    
    print("\n" + "="*60)
    if all_valid:
        print("✅ ALL TRANSFORMATIONS VALID - Ready for Step 3")
    else:
        print("⚠️  VALIDATION WARNINGS - Check output above")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
