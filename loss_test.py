import torch
from torch import nn

loss_cal = nn.CrossEntropyLoss()
pre = torch.tensor([[0.8, 0.5, 0.2, 0.5],
                    [0.2, 0.9, 0.3, 0.2],
                    [0.4, 0.3, 0.7, 0.1],
                    [0.1, 0.2, 0.4, 0.8]], dtype=torch.float)
target = torch.tensor([0, 1, 2, 3], dtype=torch.long)
print(loss_cal(pre, target))
