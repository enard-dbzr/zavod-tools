import torch
from torch import nn


class BatchNormW(nn.BatchNorm1d):
    def forward(self, x):
        bn = super()
        return torch.permute(bn.forward(torch.permute(x, (0, 2, 1))), (0, 2, 1))
