import torch
from torch import Tensor


def diff_matrix(t: Tensor):
    """Constructs a difference matrix, where d_ij correspond to x_i - x_j of the batch."""
    # Expand last dim
    if t.shape[-1] != 1:
        t = t.unsqueeze(1)
    t_expanded = t.expand(
        -1, len(t), -1
    )  # Shape: (batch_size, batch_size, tensor_size)
    return t_expanded - t_expanded.transpose(0, 1)


def diff_distance(diff_m: Tensor, p=2):
    """Calculates the norm for each entry of a difference matrix. Defaults to l2-norm."""
    return torch.linalg.vector_norm(diff_m, ord=p, dim=-1)
