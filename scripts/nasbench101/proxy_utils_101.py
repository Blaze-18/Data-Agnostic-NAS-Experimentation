"""
Shared utilities for NAS-Bench-101 proxy computation.

Builds a PyTorch nn.Module from a NAS-Bench-101 DAG specification:
  - adjacency: (7,7) upper-triangular int8 numpy array; A[i,j]=1 means edge i->j
  - operations: list of 7 strings; ops[0]='input', ops[6]='output',
                ops[1-5] in {'conv3x3-bn-relu', 'conv1x1-bn-relu', 'maxpool3x3'}

Network structure (matching the NAS-Bench-101 paper, Ying et al. 2019):
  stem -> stack0 (3 cells, C ch) -> downsample -> stack1 (3 cells, 2C ch)
       -> downsample -> stack2 (3 cells, 4C ch) -> avgpool -> FC(10)

For proxy computation we use C=16 (small, fast). The actual benchmark uses C=128.
param_count proxy is NOT derived from this model -- it is read directly from
the benchmark's fixed["trainable_parameters"] and saved in step0_audit_101.py.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Operations within a cell (no stride, no channel change -- spatial dims fixed)
# ---------------------------------------------------------------------------

def build_op(op_name: str, C: int) -> nn.Module:
    """Build a single operation module. All ops preserve spatial dimensions."""
    if op_name == "conv3x3-bn-relu":
        return nn.Sequential(
            nn.Conv2d(C, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
    elif op_name == "conv1x1-bn-relu":
        return nn.Sequential(
            nn.Conv2d(C, C, 1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )
    elif op_name == "maxpool3x3":
        # MaxPool preserves spatial dimensions (stride=1, padding=1) within a cell
        # No BN or ReLU -- this matches the original benchmark definition
        return nn.MaxPool2d(3, stride=1, padding=1)
    else:
        raise ValueError(f"Unknown op: {op_name}")


# ---------------------------------------------------------------------------
# Single NAS-Bench-101 cell (7-node DAG)
# ---------------------------------------------------------------------------

class NASBench101Cell(nn.Module):
    """
    One NAS-Bench-101 cell.

    DAG execution:
      - node 0 (INPUT): receives the input tensor
      - nodes 1-5 (intermediate): output = op(sum of all predecessor outputs)
      - node 6 (OUTPUT): sum of all predecessor outputs (no op)
    Nodes with no predecessors that reach the output contribute zero (pruned).
    """

    def __init__(self, adjacency: np.ndarray, operations: List[str], C: int):
        super().__init__()
        N = adjacency.shape[0]  # number of nodes (4-7, variable)
        assert adjacency.shape == (N, N), "adjacency must be square"
        assert len(operations) == N

        self.adj = adjacency.astype(np.int8)
        self.C   = C
        self.N   = N

        # Build op modules for intermediate nodes 1 to N-2
        # (node 0 = input, node N-1 = output)
        self.ops = nn.ModuleDict()
        for j in range(1, N - 1):
            has_predecessor = bool(self.adj[:j, j].any())
            if has_predecessor:
                self.ops[str(j)] = build_op(operations[j], C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = self.N
        node_out = [None] * N
        node_out[0] = x

        for j in range(1, N):
            # Collect outputs of all predecessors i < j with adj[i,j]=1
            preds = [node_out[i] for i in range(j)
                     if self.adj[i, j] == 1 and node_out[i] is not None]

            if not preds:
                node_out[j] = torch.zeros_like(x)
                continue

            agg = preds[0]
            for p in preds[1:]:
                agg = agg + p

            if j == N - 1:
                # Output node -- no op applied
                node_out[j] = agg
            else:
                key = str(j)
                if key in self.ops:
                    node_out[j] = self.ops[key](agg)
                else:
                    node_out[j] = agg

        return node_out[N - 1]


# ---------------------------------------------------------------------------
# Downsampling layer between stacks (doubles channels, halves spatial)
# ---------------------------------------------------------------------------

class Downsample(nn.Module):
    def __init__(self, C_in: int, C_out: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.proj = nn.Conv2d(C_in, C_out, 1, bias=False)
        self.bn   = nn.BatchNorm2d(C_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.proj(self.pool(x)))


# ---------------------------------------------------------------------------
# Full NAS-Bench-101 network
# ---------------------------------------------------------------------------

class NASBench101Net(nn.Module):
    """
    Full NAS-Bench-101 network for zero-cost proxy computation.

    Architecture (from the paper):
      stem (3->C) -> [stack0: 3 cells at C ch] -> downsample (C->2C)
                  -> [stack1: 3 cells at 2C ch] -> downsample (2C->4C)
                  -> [stack2: 3 cells at 4C ch] -> global avgpool -> FC(10)

    C=16 by default (fast proxy computation; actual benchmark uses C=128).
    """

    NUM_STACKS           = 3
    NUM_CELLS_PER_STACK  = 3

    def __init__(self, adjacency: np.ndarray, operations: List[str], C: int = 16):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C),
            nn.ReLU(inplace=True),
        )

        self.stacks       = nn.ModuleList()
        self.downsamplers = nn.ModuleList()

        current_C = C
        for s in range(self.NUM_STACKS):
            cells = nn.ModuleList([
                NASBench101Cell(adjacency, operations, current_C)
                for _ in range(self.NUM_CELLS_PER_STACK)
            ])
            self.stacks.append(cells)

            if s < self.NUM_STACKS - 1:
                next_C = current_C * 2
                self.downsamplers.append(Downsample(current_C, next_C))
                current_C = next_C

        self.avgpool    = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(current_C, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for s, cells in enumerate(self.stacks):
            for cell in cells:
                x = cell(x)
            if s < len(self.downsamplers):
                x = self.downsamplers[s](x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ---------------------------------------------------------------------------
# Convenience builder and device helper
# ---------------------------------------------------------------------------

def build_nasbench101_model(adjacency: np.ndarray,
                            operations: List[str],
                            C: int = 16) -> nn.Module:
    """
    Build and return a NASBench101Net in eval mode on CPU.

    Args:
        adjacency:  (7,7) int8 numpy array from fixed["module_adjacency"]
        operations: list of 7 strings from fixed["module_operations"]
        C:          base channel count (16 for proxy computation)

    Returns:
        nn.Module in eval mode on CPU
    """
    model = NASBench101Net(adjacency, operations, C=C)
    model.eval()
    return model


def get_device() -> torch.device:
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

    from nasbench import api
    nb = api.NASBench("data/nasbench101/nasbench_full.tfrecord")

    import random
    random.seed(0)
    hashes = list(nb.hash_iterator())
    sample = random.sample(hashes, 5)

    print("Testing NASBench101Net on 5 random architectures ...\n")
    all_ok = True
    for h in sample:
        fixed, _ = nb.get_metrics_from_hash(h)
        adj  = fixed["module_adjacency"]
        ops  = fixed["module_operations"]
        model = build_nasbench101_model(adj, ops, C=16)

        x = torch.ones(1, 3, 32, 32)
        try:
            out = model(x)
            shape_ok = out.shape == (1, 10)
            print(f"  hash={h[:8]}  output shape={out.shape}  {'OK' if shape_ok else 'FAIL'}")
            if not shape_ok:
                all_ok = False
        except Exception as e:
            print(f"  hash={h[:8]}  EXCEPTION: {e}")
            all_ok = False

    print(f"\nAll forward passes {'PASSED' if all_ok else 'FAILED'}")
