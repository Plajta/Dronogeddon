import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(DEVICE)

print(torch.cuda.memory_summary(device=None, abbreviated=False))

