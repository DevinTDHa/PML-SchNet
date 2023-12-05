import torch
from torch import Tensor
from torch import nn


class ShiftedSoftPlus(nn.Module):
    @staticmethod
    def forward(x: Tensor):
        return torch.log(0.5 * torch.exp(x) + 0.5)
