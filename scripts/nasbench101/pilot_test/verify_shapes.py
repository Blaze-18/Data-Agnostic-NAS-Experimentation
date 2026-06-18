import pickle, numpy as np, torch, sys
sys.path.insert(0, 'scripts/nasbench101')
from proxy_utils_101 import build_nasbench101_model

with open('results/nasbench101/audit/arch_specs.pkl','rb') as f:
    specs = pickle.load(f)

hashes = list(specs.keys())
test_hashes = []
for target_shape in [(4,4),(5,5),(6,6),(7,7)]:
    for h in hashes:
        if specs[h]['adjacency'].shape == target_shape:
            test_hashes.append(h)
            break

for h in test_hashes:
    s = specs[h]
    adj_shape = s['adjacency'].shape
    try:
        m = build_nasbench101_model(s['adjacency'], s['ops'], C=16)
        out = m(torch.ones(1,3,32,32))
        print("adj_shape=" + str(adj_shape) + "  out=" + str(tuple(out.shape)) + "  OK")
    except Exception as e:
        print("adj_shape=" + str(adj_shape) + "  FAIL: " + str(e))
