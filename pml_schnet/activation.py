import torch
from torch import Tensor
from torch import nn
from torch.nn import ReLU


class ShiftedSoftPlus(nn.Module):
    """Shifted Softplus version, that reverts to linear growth after a certain threshold for numerical stability.

    Seems like this one doesn't work that well either, as some terms of the gradients explode and cause
    numerical unstability.
    """

    def __init__(self, threshold=20.0):
        super().__init__()
        self.threshold: Tensor = torch.tensor(threshold)

    def forward(self, x: Tensor):
        apply_threshold = x.ge(self.threshold)
        ssp = torch.log(0.5 * torch.exp(x) + 0.5)
        result = torch.where(apply_threshold, x, ssp)
        return result


class ShiftedSoftPlusOld(nn.Module):
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
