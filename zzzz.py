import torch

x = torch.randint(-1, 1, (1, 10))
y = torch.randint(-1, 1, (1, 10))
print(x == y)
