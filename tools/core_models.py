import torch
from torch import nn
import torch.nn.functional as F


class BatchNormW(nn.BatchNorm1d):
    def forward(self, x):
        bn = super()
        return torch.permute(bn.forward(torch.permute(x, (0, 2, 1))), (0, 2, 1))


class SingleShotAttention(nn.Module):
    def __init__(self, in_features, layers=2):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, in_features),
        )

        for _ in range(layers - 1):
            self.fc1.extend([
                nn.ReLU(),
                nn.Linear(in_features, in_features),
            ])

    def forward(self, x):
        return x * F.sigmoid(self.fc1(x))
