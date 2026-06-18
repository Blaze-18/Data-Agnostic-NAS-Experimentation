"""proxy_utils_sss.py — NATS-Bench SSS model builder (DynamicShapeTinyNet).

The SSS (Size Search Space) in NATS-Bench searches over channel widths while
keeping the cell topology fixed.  Every architecture uses the same genotype:

    |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+|skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|

Arch string format:  'ch0:ch1:ch2:ch3:ch4'
  ch0  — stem output channels  (out_channel_of_1st_conv_layer)
  ch1  — 1st ResBlock output   (out_channel_of_1st_cell_stage)
  ch2  — InferCell[1] output   (out_channel_of_1st_residual_block)
  ch3  — 2nd ResBlock output   (out_channel_of_2nd_cell_stage)
  ch4  — final feature channels (out_channel_of_2nd_residual_block)

Macro skeleton (N=1 stage, DynamicShapeTinyNet):

    stem            Conv2d(3 → ch0, 3×3) + BN
    cell[0]         SSSInferCell(ch0 → ch0)          non-reduction
    cell[1]         ResNetBasicblock(ch0 → ch1, s=2)  stride-2 reduction
    cell[2]         SSSInferCell(ch1 → ch2)          non-reduction
    cell[3]         ResNetBasicblock(ch2 → ch3, s=2)  stride-2 reduction
    cell[4]         SSSInferCell(ch3 → ch4)          non-reduction
    lastact         BN(ch4) + ReLU
    global_pooling  AdaptiveAvgPool2d(1)
    classifier      Linear(ch4 → 10)

Architecture verified against stored benchmark param counts:
  '8:8:8:8:8'          →  11,714 params  (stored 0.011714 M)  ✓
  '8:8:16:40:40'        → 107,826 params  (stored 0.107826 M)  ✓
  '64:64:64:64:64'      → 713,674 params  (stored 0.713674 M)  ✓

Source:  AutoDL-Projects / xautodl / models / shape_infers / InferTinyCellNet.py
         AutoDL-Projects / xautodl / models / cell_operations.py
         AutoDL-Projects / xautodl / models / cell_infers / cells.py
"""

import torch
import torch.nn as nn


# ─── Building blocks ────────────────────────────────────────────────────────

class ReLUConvBN(nn.Module):
    """ReLU → Conv2d → BN  (the 'nor_conv_3x3' operation in NATS-Bench)."""

    def __init__(self, C_in, C_out, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=True),
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):
    """1×1 Conv + BN projection — skip_connect when C_in ≠ C_out (stride=1)."""

    def __init__(self, C_in, C_out):
        super().__init__()
        self.conv = nn.Conv2d(C_in, C_out, kernel_size=1,
                              stride=1, padding=0, bias=False)
        self.bn   = nn.BatchNorm2d(C_out, affine=True)

    def forward(self, x):
        return self.bn(self.conv(x))


class SSSInferCell(nn.Module):
    """
    Fixed-topology NAS-Bench-201 cell for NATS-Bench SSS (stride always 1).

    Genotype (shared by ALL SSS architectures):
        |nor_conv_3x3~0|+|nor_conv_3x3~0|nor_conv_3x3~1|+
        |skip_connect~0|nor_conv_3x3~1|nor_conv_3x3~2|

    4-node DAG (nodes 0–3):
        x0  = cell input
        x1  = op_0_1(x0)                                 [nor_conv_3x3, C_in→C_out]
        x2  = op_0_2(x0) + op_1_2(x1)                   [both nor_conv_3x3]
        x3  = skip_0_3(x0) + op_1_3(x1) + op_2_3(x2)   [skip + 2×nor_conv_3x3]
        out = x3

    skip_connect: Identity if C_in==C_out, else FactorizedReduce(C_in, C_out).
    Edges from input node (op_in=0) use C_in → C_out.
    Edges from intermediate nodes use C_out → C_out.
    """

    def __init__(self, C_in, C_out):
        super().__init__()
        self.in_dim  = C_in
        self.out_dim = C_out

        # Node 1: one incoming edge from node 0
        self.op_0_1 = ReLUConvBN(C_in,  C_out)   # 0 → 1

        # Node 2: two incoming edges
        self.op_0_2 = ReLUConvBN(C_in,  C_out)   # 0 → 2
        self.op_1_2 = ReLUConvBN(C_out, C_out)   # 1 → 2

        # Node 3: three incoming edges
        if C_in == C_out:
            self.skip_0_3 = nn.Identity()                # 0 → 3  (no params)
        else:
            self.skip_0_3 = FactorizedReduce(C_in, C_out)  # 0 → 3
        self.op_1_3 = ReLUConvBN(C_out, C_out)   # 1 → 3
        self.op_2_3 = ReLUConvBN(C_out, C_out)   # 2 → 3

    def forward(self, x):
        x1 = self.op_0_1(x)
        x2 = self.op_0_2(x) + self.op_1_2(x1)
        x3 = self.skip_0_3(x) + self.op_1_3(x1) + self.op_2_3(x2)
        return x3


class SSSResBlock(nn.Module):
    """
    ResNet basic block used as stride-2 reduction in DynamicShapeTinyNet.

    Structure (from cell_operations.ResNetBasicblock, stride=2):
        conv_a    : ReLUConvBN(C_in → C_out, 3×3, stride=2)
        conv_b    : ReLUConvBN(C_out → C_out, 3×3)
        downsample: AvgPool2d(2,2) + Conv2d(C_in → C_out, 1×1, stride=1, bias=False)
        out = conv_b(conv_a(x)) + downsample(x)
    """

    def __init__(self, C_in, C_out):
        super().__init__()
        self.in_dim  = C_in
        self.out_dim = C_out
        self.conv_a  = ReLUConvBN(C_in,  C_out, stride=2)
        self.conv_b  = ReLUConvBN(C_out, C_out)
        self.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(C_in, C_out, kernel_size=1,
                      stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        out      = self.conv_b(self.conv_a(x))
        residual = self.downsample(x)
        return out + residual


# ─── Full network ────────────────────────────────────────────────────────────

class SSSNet(nn.Module):
    """
    Exact DynamicShapeTinyNet replica for NATS-Bench SSS proxy computation.

    Parameters
    ----------
    channels : list[int], length 5
        [ch0, ch1, ch2, ch3, ch4] parsed from arch_str 'a:b:c:d:e'.
    num_classes : int
        Number of output classes (10 for CIFAR-10).
    """

    def __init__(self, channels, num_classes=10):
        super().__init__()
        if len(channels) != 5:
            raise ValueError(
                "SSS requires exactly 5 channel values, got %d" % len(channels))
        ch0, ch1, ch2, ch3, ch4 = channels

        self.stem = nn.Sequential(
            nn.Conv2d(3, ch0, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ch0),
        )

        self.cells = nn.ModuleList([
            SSSInferCell(ch0, ch0),     # cell[0]: non-reduction
            SSSResBlock(ch0, ch1),      # cell[1]: stride-2 reduction
            SSSInferCell(ch1, ch2),     # cell[2]: non-reduction
            SSSResBlock(ch2, ch3),      # cell[3]: stride-2 reduction
            SSSInferCell(ch3, ch4),     # cell[4]: non-reduction
        ])

        self.lastact = nn.Sequential(
            nn.BatchNorm2d(ch4),
            nn.ReLU(inplace=True),
        )
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(ch4, num_classes)

    def forward(self, x):
        out = self.stem(x)
        for cell in self.cells:
            out = cell(out)
        out = self.lastact(out)
        out = self.global_pooling(out)
        out = out.view(out.size(0), -1)
        logits = self.classifier(out)
        return out, logits


# ─── Public helpers ──────────────────────────────────────────────────────────

def parse_arch_str(arch_str):
    """Parse SSS arch string 'ch0:ch1:ch2:ch3:ch4' → list of 5 ints."""
    parts = arch_str.strip().split(':')
    if len(parts) != 5:
        raise ValueError(
            "SSS arch_str must have 5 colon-separated values, got %r" % arch_str)
    return [int(c) for c in parts]


def build_sss_model(arch_str, num_classes=10):
    """Build SSSNet from arch string.  Returns CPU, eval-mode model."""
    channels = parse_arch_str(arch_str)
    model = SSSNet(channels, num_classes=num_classes)
    model.eval()
    return model
