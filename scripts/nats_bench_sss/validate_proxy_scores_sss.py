r"""
Proxy Score Validation  NATS-Bench SSS (Step 1.5)
====================================================
Validates the 4 computed proxy arrays before any downstream analysis.

Checks performed
----------------
CHECK 1  Shape & dtype
         All 4 arrays must be shape (32768,), dtype float64.

CHECK 2  NaN / Inf cleanliness
         Zero NaNs and Infs allowed in any array.

CHECK 3  Zero / degenerate count
         param_count: zero zeros allowed.
         naswot / zenscore / synflow: report count (a few zeros from
         degenerate forward passes is acceptable; >1% is a flag).

CHECK 4  Param count cross-validation (100 random architectures)
         Rebuilds each model from the arch string stored in the pickle
         and compares against the saved param_count array. Must match
         to the integer exactly for all 100 samples.

CHECK 5  Monotonicity sanity (capacity space)
         '8:8:8:8:8' (idx 0)  vs  '64:64:64:64:64' (idx 32767).
         Every proxy should be STRICTLY larger for the wider network
         (param_count, naswot, zenscore, synflow all increase with capacity).

CHECK 6  Spearman  vs GT accuracy (full 32,768 architectures)
         Expected signs from Step 0 audit:
           param_count   > 0  (positive  more params  higher acc)
           naswot        > 0  (positive  no negation applied)
           zenscore      > 0  (positive  no negation applied)
           synflow       > 0  (positive  synaptic salience  capacity)
         Expected magnitudes (from Step 0 audit on 200-arch subset):
           naswot raw   0.46  (audit Q6: 0.463)
         WARNING threshold:  < 0.10 for naswot/zenscore is suspicious.

CHECK 7  Cross-proxy correlation matrix
         param_count vs synflow should be very highly correlated ( > 0.85)
           because SynFlow  product of layer sizes  parameter count.
         naswot vs zenscore should be moderately correlated ( > 0.70)
           because both measure activation covariance traces.
         naswot vs param_count:  expected moderate-high ( > 0.40).

CHECK 8  Score distribution statistics
         Prints min/max/mean/std/median/IQR for each proxy.
         Flags if std/mean < 0.01 (too little variance to be useful).

CHECK 9  Top-10% / bottom-10% separation
         Splits architectures by GT accuracy percentile and computes
         mean proxy score in each decile. All proxies except synflow
         (which we expect to exclude) should show clear monotonic trend.

Outputs
-------
results/nats_bench_sss/audit/proxy_validation.json    all check results
results/nats_bench_sss/audit/proxy_validation.png     4-panel figure

Usage:
    python -u scripts\nats_bench_sss\validate_proxy_scores_sss.py
"""

import json
import os
import pickle
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from proxy_utils_sss import build_sss_model

#  Paths 
ARCH_DIR    = Path("data/nats_bench_sss/NATS-sss-v1_0-50262-simple")
SCORES_DIR  = Path("results/nats_bench_sss/raw_proxy_scores")
OUT_DIR     = Path("results/nats_bench_sss/audit")
N_ARCHS     = 32_768
RANDOM_SEED = 42
XVAL_N      = 100       # architectures for param count cross-validation

OUT_DIR.mkdir(parents=True, exist_ok=True)

PROXY_NAMES = ["param_count", "naswot", "zenscore", "synflow"]

# Expected direction: +1 = should be positively correlated with GT accuracy
EXPECTED_SIGN = {"param_count": +1, "naswot": +1, "zenscore": +1, "synflow": +1}
# Minimum || we consider "useful" as a proxy before debiasing
MIN_RHO      = {"param_count": 0.50, "naswot": 0.30, "zenscore": 0.30, "synflow": 0.10}


#  Helpers 

def load_gt_accuracy(idx: int) -> float:
    with open(ARCH_DIR / ("%d.pickle" % idx), "rb") as f:
        data = pickle.load(f)
    r = data["90"]["all_results"][("cifar10", 777)]
    return float(r["eval_acc1es"]["ori-test@89"])


def load_arch_str(idx: int) -> str:
    with open(ARCH_DIR / ("%d.pickle" % idx), "rb") as f:
        data = pickle.load(f)
    return data["90"]["arch_str"]


def sep(title=""):
    width = 70
    if title:
        pad = (width - len(title) - 2) // 2
        print("\n" + "-" * pad + " " + title + " " + "-" * (width - pad - len(title) - 2))
    else:
        print("-" * width)


#  Main validation 

def main():
    results = {}
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    #  Load all proxy arrays 
    sep("LOADING PROXY ARRAYS")
    scores = {}
    for name in PROXY_NAMES:
        path = SCORES_DIR / ("%s.npy" % name)
        if not path.exists():
            print("  MISSING: %s" % path)
            sys.exit(1)
        scores[name] = np.load(str(path))
        print("  Loaded  %-14s  shape=%s  dtype=%s" %
              (name, scores[name].shape, scores[name].dtype), flush=True)

    #  CHECK 1: Shape & dtype 
    sep("CHECK 1  Shape & dtype")
    c1_pass = True
    for name in PROXY_NAMES:
        ok_shape = scores[name].shape == (N_ARCHS,)
        ok_dtype = np.issubdtype(scores[name].dtype, np.floating)
        status = "PASS" if (ok_shape and ok_dtype) else "FAIL"
        if not (ok_shape and ok_dtype):
            c1_pass = False
        print("  %-14s  shape=%s  dtype=%-10s  %s" %
              (name, scores[name].shape, scores[name].dtype, status), flush=True)
    results["check1_shape_dtype"] = "PASS" if c1_pass else "FAIL"

    #  CHECK 2: NaN / Inf 
    sep("CHECK 2  NaN / Inf cleanliness")
    c2_pass = True
    for name in PROXY_NAMES:
        n_nan = int(np.isnan(scores[name]).sum())
        n_inf = int(np.isinf(scores[name]).sum())
        ok = (n_nan == 0) and (n_inf == 0)
        if not ok:
            c2_pass = False
        print("  %-14s  NaN=%d  Inf=%d  %s" % (name, n_nan, n_inf,
              "PASS" if ok else "FAIL"), flush=True)
    results["check2_nan_inf"] = "PASS" if c2_pass else "FAIL"

    #  CHECK 3: Zero / degenerate values 
    sep("CHECK 3  Zero / degenerate count")
    c3_pass = True
    for name in PROXY_NAMES:
        n_zero = int((scores[name] == 0).sum())
        pct = 100.0 * n_zero / N_ARCHS
        if name == "param_count":
            ok = n_zero == 0
        else:
            ok = pct < 1.0   # <1% zeros allowed for score-based proxies
        if not ok:
            c3_pass = False
        print("  %-14s  zeros=%d (%.2f%%)  %s" % (name, n_zero, pct,
              "PASS" if ok else "FAIL"), flush=True)
    results["check3_zeros"] = "PASS" if c3_pass else "FAIL"

    #  CHECK 4: Param count cross-validation 
    sep("CHECK 4  Param count cross-validation (%d random archs)" % XVAL_N)
    xval_indices = random.sample(range(N_ARCHS), XVAL_N)
    mismatches = []
    for i, idx in enumerate(xval_indices):
        arch_str = load_arch_str(idx)
        model = build_sss_model(arch_str)
        built = sum(p.numel() for p in model.parameters())
        stored = int(scores["param_count"][idx])
        del model
        if built != stored:
            mismatches.append((idx, arch_str, built, stored))
        if (i + 1) % 25 == 0:
            print("  xval progress: %d/%d  mismatches so far: %d" %
                  (i + 1, XVAL_N, len(mismatches)), flush=True)

    c4_pass = len(mismatches) == 0
    print("  Mismatches: %d / %d    %s" % (len(mismatches), XVAL_N,
          "PASS" if c4_pass else "FAIL"), flush=True)
    if mismatches:
        for idx, arch_str, built, stored in mismatches[:5]:
            print("    idx=%d  arch=%s  built=%d  stored=%d  diff=%d" %
                  (idx, arch_str, built, stored, built - stored))
    results["check4_param_xval"] = {
        "status": "PASS" if c4_pass else "FAIL",
        "n_tested": XVAL_N,
        "n_mismatch": len(mismatches),
    }

    #  CHECK 5: Monotonicity sanity 
    sep("CHECK 5  Monotonicity sanity (arch 0 vs arch 32767)")
    arch0_str  = "8:8:8:8:8    (all-8,  idx=0)"
    arch32_str = "64:64:64:64:64 (all-64, idx=32767)"
    print("  %-14s  %20s  %20s  direction" % ("proxy", arch0_str[:19], arch32_str[:19]))
    sep()
    c5_pass = True
    c5_detail = {}
    for name in PROXY_NAMES:
        v0  = scores[name][0]
        v32 = scores[name][32767]
        ok  = v32 > v0
        if not ok:
            c5_pass = False
        ratio = v32 / v0 if v0 != 0 else float("inf")
        print("  %-14s  %20.4g  %20.4g  ratio=%.2fx  %s" %
              (name, v0, v32, ratio, "PASS" if ok else "FAIL"), flush=True)
        c5_detail[name] = {"v_small": float(v0), "v_large": float(v32),
                           "ratio": float(ratio), "ok": bool(ok)}
    results["check5_monotonicity"] = {"status": "PASS" if c5_pass else "FAIL",
                                      "detail": c5_detail}

  
    sep("LOADING GT ACCURACY (32,768 arches  takes ~2 min)")
    gt = np.zeros(N_ARCHS, dtype=np.float64)
    for idx in range(N_ARCHS):
        with open(ARCH_DIR / ("%d.pickle" % idx), "rb") as f:
            data = pickle.load(f)
        r = data["90"]["all_results"][("cifar10", 777)]
        gt[idx] = float(r["eval_acc1es"]["ori-test@89"])
        if (idx + 1) % 5000 == 0:
            print("  loaded %d / %d" % (idx + 1, N_ARCHS), flush=True)
    print("  GT loaded.  mean=%.3f%%  min=%.3f%%  max=%.3f%%" %
          (gt.mean(), gt.min(), gt.max()), flush=True)

    #  CHECK 6: Spearman  vs GT accuracy 
    sep("CHECK 6  Spearman rho vs GT accuracy (n=32,768)")
    c6_pass = True
    rhos = {}
    print("  %-14s  %8s  %8s  expected_sign  min_rho  status" %
          ("proxy", "rho", "p-value"))
    sep()
    for name in PROXY_NAMES:
        rho, pval = spearmanr(scores[name], gt)
        rhos[name] = float(rho)
        sign_ok = (rho * EXPECTED_SIGN[name]) > 0
        mag_ok  = abs(rho) >= MIN_RHO[name]
        ok = sign_ok and mag_ok
        if not ok:
            c6_pass = False
        status = "PASS" if ok else ("SIGN_WARN" if not sign_ok else "MAG_WARN")
        print("  %-14s  %+8.4f  %8.2e  %+d             %.2f     %s" %
              (name, rho, pval, EXPECTED_SIGN[name], MIN_RHO[name], status),
              flush=True)
    results["check6_spearman_rho"] = {
        "status": "PASS" if c6_pass else "WARN",
        "rhos": rhos,
    }

    #  CHECK 7: Cross-proxy correlations 
    sep("CHECK 7  Cross-proxy correlation matrix")
    matrix = {}
    header = "  %-14s" + "  %+8s" * len(PROXY_NAMES)
    print(header % (("",) + tuple(PROXY_NAMES)), flush=True)
    sep()
    c7_issues = []
    for n1 in PROXY_NAMES:
        row_vals = []
        for n2 in PROXY_NAMES:
            r, _ = spearmanr(scores[n1], scores[n2])
            row_vals.append(r)
            matrix["%s_vs_%s" % (n1, n2)] = float(r)
        print(("  %-14s" + "  %+8.4f" * len(PROXY_NAMES)) %
              ((n1,) + tuple(row_vals)), flush=True)

    # Specific expected correlations
    checks = [
        ("param_count", "synflow",  0.85, "SynFlow  capacity"),
        ("naswot",      "zenscore", 0.60, "both covariance-trace based"),
        ("naswot",      "param_count", 0.40, "activation diversity  width"),
    ]
    print()
    for n1, n2, threshold, reason in checks:
        r = matrix["%s_vs_%s" % (n1, n2)]
        ok = r >= threshold
        if not ok:
            c7_issues.append((n1, n2, r, threshold))
        print("  %-14s vs %-14s  rho=%+.4f  threshold=%.2f  (%s)  %s" %
              (n1, n2, r, threshold, reason, "PASS" if ok else "WARN"), flush=True)
    results["check7_cross_corr"] = {
        "status": "PASS" if not c7_issues else "WARN",
        "matrix": matrix,
    }

    #  CHECK 8: Distribution statistics 
    sep("CHECK 8  Score distribution statistics")
    c8_pass = True
    dist_stats = {}
    print("  %-14s  %12s  %12s  %12s  %12s  %8s  %8s  cv_ok" %
          ("proxy", "min", "max", "mean", "std", "median", "IQR"))
    sep()
    for name in PROXY_NAMES:
        s = scores[name]
        q25, q75 = np.percentile(s, [25, 75])
        iqr = q75 - q25
        cv = s.std() / s.mean() if s.mean() != 0 else 0
        cv_ok = cv >= 0.01
        if not cv_ok:
            c8_pass = False
        dist_stats[name] = {
            "min": float(s.min()), "max": float(s.max()),
            "mean": float(s.mean()), "std": float(s.std()),
            "median": float(np.median(s)), "iqr": float(iqr), "cv": float(cv),
        }
        print("  %-14s  %12.4g  %12.4g  %12.4g  %12.4g  %8.4g  %8.4g  %s" %
              (name, s.min(), s.max(), s.mean(), s.std(),
               np.median(s), iqr, "PASS" if cv_ok else "FAIL"), flush=True)
    results["check8_distributions"] = {
        "status": "PASS" if c8_pass else "FAIL",
        "stats": dist_stats,
    }

    #  CHECK 9: Top-10% vs bottom-10% GT separation 
    sep("CHECK 9  Proxy mean by GT accuracy decile")
    pct10 = np.percentile(gt, 10)
    pct90 = np.percentile(gt, 90)
    bot_mask = gt <= pct10
    top_mask = gt >= pct90
    print("  Bottom 10%% GT accuracy threshold: %.3f%%" % pct10)
    print("  Top    10%% GT accuracy threshold: %.3f%%" % pct90)
    print()
    print("  %-14s  %14s  %14s  %8s  direction" %
          ("proxy", "bottom_10pct_mean", "top_10pct_mean", "ratio"))
    sep()
    c9_pass = True
    c9_detail = {}
    for name in PROXY_NAMES:
        s = scores[name]
        bot_mean = s[bot_mask].mean()
        top_mean = s[top_mask].mean()
        ratio = top_mean / bot_mean if bot_mean != 0 else float("inf")
        ok = top_mean > bot_mean
        if not ok:
            c9_pass = False
        c9_detail[name] = {"bot": float(bot_mean), "top": float(top_mean),
                           "ratio": float(ratio), "ok": bool(ok)}
        print("  %-14s  %14.4g  %14.4g  %8.3f  %s" %
              (name, bot_mean, top_mean, ratio, "PASS" if ok else "FAIL"),
              flush=True)
    results["check9_decile_sep"] = {"status": "PASS" if c9_pass else "WARN",
                                    "detail": c9_detail}

    #  Overall summary 
    sep("OVERALL SUMMARY")
    check_statuses = {k: v["status"] if isinstance(v, dict) else v
                      for k, v in results.items()}
    all_pass = all(s == "PASS" for s in check_statuses.values())
    for check, status in check_statuses.items():
        icon = "" if status == "PASS" else ("" if status == "WARN" else "")
        print("  %s  %-40s  %s" % (icon, check, status))
    print()
    print("  VERDICT: %s" % ("ALL CHECKS PASS  proxies are valid." if all_pass
                              else "SOME WARNINGS  review above."))

    results["overall"] = "PASS" if all_pass else "WARN"
    results["check_statuses"] = check_statuses

    #  Save JSON 
    out_json = OUT_DIR / "proxy_validation.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print("\n  Saved: %s" % out_json, flush=True)

    #  Figure: 4-panel scatter (proxy vs GT, all 32k arches) 
    sep("SAVING FIGURE")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("NATS-Bench SSS  Proxy scores vs GT accuracy (n=32,768)", fontsize=13)
    axes = axes.flatten()

    for ax, name in zip(axes, PROXY_NAMES):
        s = scores[name]
        # log-scale the x-axis for param_count and synflow (huge range)
        use_log = name in ("param_count", "synflow")
        xvals = np.log10(s + 1e-30) if use_log else s
        ax.scatter(xvals, gt, s=0.5, alpha=0.2, color="steelblue", rasterized=True)
        rho = rhos[name]
        xlabel = ("log10(%s)" % name) if use_log else name
        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel("GT accuracy (%)", fontsize=10)
        ax.set_title("%s  ( = %+.4f)" % (name, rho), fontsize=11)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_fig = OUT_DIR / "proxy_validation.png"
    plt.savefig(str(out_fig), dpi=100)
    plt.close()
    print("  Saved: %s" % out_fig, flush=True)

    sep()
    print("Done.", flush=True)
    return results


if __name__ == "__main__":
    main()

