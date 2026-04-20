#!/usr/bin/env python3
"""
Test script: build 16 random small CNNs, run them on DirectML device,
compute an activation-variance proxy (Zen-like) and parameter count.

Run:
  python scripts/test/compute_proxy_directml_test.py
"""
import random
import torch
import torch.nn as nn
import torch_directml

device = torch_directml.device()
print('Using device:', device)


def build_random_cnn(seed=None):
    rnd = random.Random(seed)
    layers = []
    in_ch = 3
    num_blocks = rnd.randint(2, 4)
    for b in range(num_blocks):
        out_ch = rnd.choice([8, 16, 32])
        layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=False))
        if rnd.random() < 0.5:
            layers.append(nn.MaxPool2d(2))
        in_ch = out_ch
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(in_ch, 1))
    return nn.Sequential(*layers)


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def activation_variance_proxy(model, device):
    # Register hooks to capture activations after ReLU
    acts = []
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            def fn(module, input, output):
                # output is a tensor
                acts.append(output.detach())
            hooks.append(m.register_forward_hook(fn))
    model.eval()
    bs = 8
    inp = torch.randn(bs, 3, 32, 32, device=device)
    try:
        model.to(device)
        with torch.no_grad():
            _ = model(inp)
    finally:
        for h in hooks:
            h.remove()
    if len(acts) == 0:
        return None
    # compute mean variance across collected activations
    var_vals = []
    for a in acts:
        # move to cpu for stats
        a_cpu = a.cpu()
        # variance across batch and spatial dims, keep channels
        dims = tuple(range(0, a_cpu.dim()))
        var = float(a_cpu.var().item())
        var_vals.append(var)
    return float(sum(var_vals) / len(var_vals))


if __name__ == '__main__':
    results = []
    for i in range(16):
        m = build_random_cnn(seed=i)
        pc = param_count(m)
        try:
            proxy = activation_variance_proxy(m, device)
        except Exception as e:
            proxy = None
            print(f"Model {i}: error during forward on device: {e}")
        results.append((i, pc, proxy))
    print('\nResults (arch_id, param_count, activation_variance_proxy):')
    for r in results:
        print(r)
