"""Quick spot-check to verify the NAS-Bench-101 data is reading correctly."""
import os, random, numpy as np
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from nasbench import api
from scipy.stats import spearmanr

nb = api.NASBench("data/nasbench101/nasbench_full.tfrecord")
hashes = list(nb.hash_iterator())
print(f"Total unique hashes: {len(hashes)}")

# 1. Spot-check 5 architectures
random.seed(42)
sample = [hashes[0], hashes[-1]] + random.sample(hashes, 3)
print("\n--- Spot-check 5 architectures ---")
for h in sample:
    fixed, computed = nb.get_metrics_from_hash(h)
    runs = computed[108]
    accs = [r["final_test_accuracy"] * 100 for r in runs]
    print(f"  hash={h[:8]}  params={fixed['trainable_parameters']:,}  "
          f"n_seeds={len(runs)}  test_accs={[round(a,3) for a in accs]}  "
          f"mean={round(sum(accs)/len(accs),3)}%")

# 2. Verify key choice: final_test_accuracy vs final_validation_accuracy
print("\n--- Key verification (test vs validation) ---")
h = hashes[100]
fixed, computed = nb.get_metrics_from_hash(h)
runs = computed[108]
for i, r in enumerate(runs):
    print(f"  seed {i}: test={r['final_test_accuracy']*100:.3f}%  "
          f"val={r['final_validation_accuracy']*100:.3f}%  "
          f"train={r['final_train_accuracy']*100:.3f}%")

# 3. Sample 1000 archs and recompute R^2 to confirm
print("\n--- R^2 recheck on 1000-arch sample ---")
random.seed(0)
s1000 = random.sample(hashes, 1000)
params_s, accs_s = [], []
for h in s1000:
    fixed, computed = nb.get_metrics_from_hash(h)
    runs = computed[108]
    params_s.append(fixed["trainable_parameters"])
    accs_s.append(np.mean([r["final_test_accuracy"] for r in runs]) * 100)
params_s = np.array(params_s)
accs_s = np.array(accs_s)
from scipy import stats as scipy_stats
_, _, r, _, _ = scipy_stats.linregress(np.log(params_s), accs_s)
rho, _ = spearmanr(params_s, accs_s)
print(f"  R^2 = {r**2:.4f}  (full run gave 0.0471 -- should be close)")
print(f"  Spearman rho = {rho:.4f}  (full run gave 0.435 -- should be close)")
print(f"  GT mean={accs_s.mean():.2f}%  std={accs_s.std():.2f}%")

print("\nVerification complete.")
