import torch
from torch import Tensor
from torch import nn
from torch.nn import Softplus, ReLU


class ShiftedSoftPlus(nn.Module):
    """Shifted Softplus version, that reverts to linear growth after a certain threshold for numerical stability."""

    def __init__(self):
        super().__init__()
        self.log_one_half = torch.log(torch.tensor(0.5))
        self.softplus = Softplus()

    def forward(self, x: Tensor):
        return self.log_one_half + self.softplus(x)


class ShiftedSoftPlusLSE(nn.Module):
    """Numerically Stable shfited softplus, Using Log Sum Exp Trick.

    But I guess not numerically stable enough...
    TODO: We always subtract the max, leading to exp(-10000000) = 0. Maybe apply to x only, where its positive.
    """

    def __init__(self):
        super().__init__()
        self.relu = ReLU()
        self.log_one_half = torch.log(torch.tensor(0.5))

    def forward(self, x: Tensor):
        x_max = self.relu(x.max())
        a = self.log_one_half + x - x_max
        b = self.log_one_half - x_max

        ssp = x_max + torch.log(torch.exp(a) + torch.exp(b))
        return ssp
