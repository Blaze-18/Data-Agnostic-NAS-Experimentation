"""
Shared utilities for NAS-Bench-201 proxy computation.
Includes model builders and common helper functions.
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Callable, Any


class NASBench201Cell(nn.Module):
    """
    A simplified cell module that interprets NAS-Bench-201 architecture strings.
    Builds a minimal computational graph from the cell definition.
    """
    
    OPS = {
        'none': lambda C_in, C_out, S: Zero(C_in, C_out, S),
        'avg_pool_3x3': lambda C_in, C_out, S: AvgPool(C_in, C_out, S),
        'nor_conv_1x1': lambda C_in, C_out, S: NorConv(C_in, C_out, 1, S),
        'nor_conv_3x3': lambda C_in, C_out, S: NorConv(C_in, C_out, 3, S),
        'skip_connect': lambda C_in, C_out, S: Identity(C_in, C_out, S) if C_in == C_out and S == 1 else Zero(C_in, C_out, S),
    }
    
    def __init__(self, arch_str: str, C_in: int = 3, C: int = 16, num_cells: int = 5):
        """
        Build cell-based network from NAS-Bench-201 arch_str.
        arch_str format: |op~in|+|op~in|op~in|+|...
        """
        super().__init__()
        self.arch_str = arch_str
        self.C = C
        self.num_cells = num_cells
        
        # Stem: input layer
        self.stem = nn.Sequential(
            nn.Conv2d(C_in, C, 3, padding=1, bias=False),
            nn.BatchNorm2d(C)
        )
        
        # Parse and build cells
        self.cells = nn.ModuleList()
        for _ in range(num_cells):
            cell = self._build_cell(arch_str, C, C)
            self.cells.append(cell)
        
        # Global average pooling + classifier
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C, 10)  # NAS-Bench-201 uses 10 classes
    
    def _build_cell(self, arch_str: str, C_in: int, C_out: int) -> nn.Module:
        """Parse arch_str and create cell module."""
        try:
            cell_layers = nn.ModuleList()
            
            # Parse the architecture string
            # Format: |op~prev|+|op~prev|op~prev|+|op~prev|...
            parts = arch_str.split('|')
            
            for part in parts:
                if part.strip() and '+' not in part:
                    if '~' in part:
                        op_name = part.split('~')[0]
                        if op_name in self.OPS:
                            op = self.OPS[op_name](C_in, C_out, 1)
                            cell_layers.append(op)
            
            return cell_layers if len(cell_layers) > 0 else nn.Sequential(Identity(C_in, C_out, 1))
        except:
            # Fallback: simple identity if parsing fails
            return nn.Sequential(Identity(C_in, C_out, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for cell in self.cells:
            for layer in cell:
                x = layer(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Zero(nn.Module):
    """Zero operation (no connection)."""
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
    
    def forward(self, x):
        return torch.zeros(x.size(0), self.C_out, x.size(2) // self.stride, x.size(3) // self.stride, 
                          device=x.device, dtype=x.dtype)


class Identity(nn.Module):
    """Identity operation (skip connection)."""
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        if stride != 1 or C_in != C_out:
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False)
        else:
            self.conv = None
    
    def forward(self, x):
        if self.conv is None:
            return x
        return self.conv(x)


class AvgPool(nn.Module):
    """Average pooling operation."""
    def __init__(self, C_in, C_out, stride):
        super().__init__()
        self.C_in = C_in
        self.C_out = C_out
        self.stride = stride
        if stride != 1 or C_in != C_out:
            self.conv = nn.Conv2d(C_in, C_out, 1, stride=stride, bias=False)
        else:
            self.conv = None
        self.pool = nn.AvgPool2d(3, padding=1, stride=stride)
    
    def forward(self, x):
        x = self.pool(x)
        if self.conv is not None:
            x = self.conv(x)
        return x


class NorConv(nn.Module):
    """Normalized convolution operation."""
    def __init__(self, C_in, C_out, kernel_size, stride):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(C_out)
        self.relu = nn.ReLU(inplace=False)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def build_nas201_model(arch_str: str, C: int = 16, num_cells: int = 5, device: str = 'cpu') -> nn.Module:
    """
    Build a NAS-Bench-201 cell-based model from architecture string.
    
    Args:
        arch_str: NAS-Bench-201 architecture string
        C: Channel multiplier
        num_cells: Number of cells to stack
        device: Device to place model on
    
    Returns:
        PyTorch model
    """
    model = NASBench201Cell(arch_str, C_in=3, C=C, num_cells=num_cells)
    model = model.to(device)
    model.eval()
    return model


def count_parameters(model: nn.Module) -> int:
    """Count total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device_name: str = 'directml') -> torch.device:
    """
    Get torch device. Tries DirectML first, falls back to CPU.
    
    Args:
        device_name: 'directml', 'cpu', 'cuda', or 'auto'
    
    Returns:
        torch.device object
    """
    if device_name == 'directml':
        try:
            import torch_directml
            device = torch_directml.device()
            print(f"✓ Using DirectML device: {device}")
            return device
        except ImportError:
            print("⚠ torch_directml not available, falling back to CPU")
            return torch.device('cpu')
    elif device_name == 'cuda':
        if torch.cuda.is_available():
            print(f"✓ Using CUDA device")
            return torch.device('cuda')
        else:
            print("⚠ CUDA not available, using CPU")
            return torch.device('cpu')
    elif device_name == 'cpu':
        return torch.device('cpu')
    else:  # auto
        if torch.cuda.is_available():
            return torch.device('cuda')
        try:
            import torch_directml
            return torch_directml.device()
        except:
            return torch.device('cpu')
