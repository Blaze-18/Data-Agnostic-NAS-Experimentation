import torch, torch_directml
dml = torch_directml.device()
x = torch.randn(16, 3, 32, 32, device=dml)
print('device:', x.device)