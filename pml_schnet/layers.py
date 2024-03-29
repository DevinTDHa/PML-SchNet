import torch
from torch import nn, Tensor

from schnetpack.nn.scatter import scatter_add


from .settings import device


class RadialBasisFunctions(nn.Module):
    def __init__(self, rbf_min: float, rbf_max: float, n_rbf: int, gamma=10):
        """Bla bla

        Parameters
        ----------
        rbf_min : float
        rbf_max : float
        n_rbf : int
        gamma : int
        """
        super().__init__()

        self.rbf_min = rbf_min
        self.rbf_max = rbf_max
        self.n_rbf = n_rbf
        self.gamma = gamma
        self.centers = torch.linspace(rbf_min, rbf_max, n_rbf).view(1, -1).to(device)

    def forward(self, R_distances: Tensor):
        diff = R_distances[..., None] - self.centers
        return torch.exp(-self.gamma * torch.pow(diff, 2))


class CfConv(nn.Module):
    """Module for learning continuous convolutions between atoms."""

    def __init__(
        self,
        atom_embeddings_dim: int,
        rbf_min: float,
        rbf_max: float,
        n_rbf: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.rbf = RadialBasisFunctions(rbf_min, rbf_max, n_rbf)
        self.w_layers = nn.Sequential(
            nn.Linear(n_rbf, atom_embeddings_dim),
            activation(),
            nn.Linear(atom_embeddings_dim, atom_embeddings_dim),
            activation(),
        )

    def forward(self, X: Tensor, R_distances: Tensor, idx_i: Tensor, idx_j: Tensor):
        # Given:
        # 1) R_distances[i][j] = ||r_i - r_j||

        # 2) rbf, 300
        radial_basis_distances = self.rbf(R_distances)

        # 3) dense, 64
        # 4) shifted softplus
        # 5) dense, 64
        # 6) shifted softplus
        Wij = self.w_layers(radial_basis_distances)

        # continuous-filter convolution output
        # X * W
        x_j = X[idx_j]
        x_ij = x_j * Wij
        return scatter_add(x_ij, idx_i, dim_size=X.shape[0])


class CfConvReg(nn.Module):
    """Module for learning continuous convolutions between atoms."""

    def __init__(
        self,
        atom_embeddings_dim: int,
        rbf_min: float,
        rbf_max: float,
        n_rbf: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.rbf = RadialBasisFunctions(rbf_min, rbf_max, n_rbf)
        self.w_layers = nn.Sequential(
            nn.Linear(n_rbf, atom_embeddings_dim, bias=False),
            RMSNorm(atom_embeddings_dim),
            activation(),
            nn.Linear(atom_embeddings_dim, atom_embeddings_dim, bias=False),
            RMSNorm(atom_embeddings_dim),
            activation(),
        )

    def forward(self, X: Tensor, R_distances: Tensor, idx_i: Tensor, idx_j: Tensor):
        # Given:
        # 1) R_distances[i][j] = ||r_i - r_j||

        # 2) rbf, 300
        radial_basis_distances = self.rbf(R_distances)

        # 3) dense, 64
        # 4) shifted softplus
        # 5) dense, 64
        # 6) shifted softplus
        Wij = self.w_layers(radial_basis_distances)

        # continuous-filter convolution output
        # X * W
        x_j = X[idx_j]
        x_ij = x_j * Wij
        return scatter_add(x_ij, idx_i, dim_size=X.shape[0])


class CfConvDropout(nn.Module):
    """Module for learning continuous convolutions between atoms."""

    def __init__(
        self,
        atom_embeddings_dim: int,
        rbf_min: float,
        rbf_max: float,
        n_rbf: int,
        activation: nn.Module,
        dropout_p,
    ):
        super().__init__()
        self.rbf = RadialBasisFunctions(rbf_min, rbf_max, n_rbf)
        self.w_layers = nn.Sequential(
            nn.Linear(n_rbf, atom_embeddings_dim, bias=False),
            nn.Dropout(dropout_p),
            activation(),
            nn.Linear(atom_embeddings_dim, atom_embeddings_dim, bias=False),
            nn.Dropout(dropout_p),
            activation(),
        )

    def forward(self, X: Tensor, R_distances: Tensor, idx_i: Tensor, idx_j: Tensor):
        # Given:
        # 1) R_distances[i][j] = ||r_i - r_j||

        # 2) rbf, 300
        radial_basis_distances = self.rbf(R_distances)

        # 3) dense, 64
        # 4) shifted softplus
        # 5) dense, 64
        # 6) shifted softplus
        Wij = self.w_layers(radial_basis_distances)

        # continuous-filter convolution output
        # X * W
        x_j = X[idx_j]
        x_ij = x_j * Wij
        return scatter_add(x_ij, idx_i, dim_size=X.shape[0])


class CfConvBNDropout(nn.Module):
    """Module for learning continuous convolutions between atoms."""

    def __init__(
        self,
        atom_embeddings_dim: int,
        rbf_min: float,
        rbf_max: float,
        n_rbf: int,
        activation: nn.Module,
        dropout_p,
    ):
        super().__init__()
        self.rbf = RadialBasisFunctions(rbf_min, rbf_max, n_rbf)
        self.w_layers = nn.Sequential(
            nn.Linear(n_rbf, atom_embeddings_dim, bias=False),
            RMSNorm(atom_embeddings_dim),
            nn.Dropout(dropout_p),
            activation(),
            nn.Linear(atom_embeddings_dim, atom_embeddings_dim, bias=False),
            RMSNorm(atom_embeddings_dim),
            nn.Dropout(dropout_p),
            activation(),
        )

    def forward(self, X: Tensor, R_distances: Tensor, idx_i: Tensor, idx_j: Tensor):
        # Given:
        # 1) R_distances[i][j] = ||r_i - r_j||

        # 2) rbf, 300
        radial_basis_distances = self.rbf(R_distances)

        # 3) dense, 64
        # 4) shifted softplus
        # 5) dense, 64
        # 6) shifted softplus
        Wij = self.w_layers(radial_basis_distances)

        # continuous-filter convolution output
        # X * W
        x_j = X[idx_j]
        x_ij = x_j * Wij
        return scatter_add(x_ij, idx_i, dim_size=X.shape[0])


class SchNetInteraction(nn.Module):
    """SchNet interaction block for modeling inter-atomic interactions."""

    def __init__(
        self,
        atom_embeddings_dim: int,
        rbf_min: float,
        rbf_max: float,
        n_rbf: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.in_atom_wise = nn.Linear(
            atom_embeddings_dim,
            atom_embeddings_dim
            # bias=False,  # TODO: why?
        )

        self.cf_conv = CfConv(atom_embeddings_dim, rbf_min, rbf_max, n_rbf, activation)

        self.out_atom_wise = nn.Sequential(
            nn.Linear(atom_embeddings_dim, atom_embeddings_dim),
            activation(),
            nn.Linear(atom_embeddings_dim, atom_embeddings_dim),
        )

    def forward(
        self,
        X: torch.Tensor,
        R_distances: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ):
        # 1) atom-wise, 64
        X_in = self.in_atom_wise(X)

        # 2) cfconv, 64
        X_conv = self.cf_conv(X_in, R_distances, idx_i, idx_j)

        # 3) atom-wise, 64
        # 4) shifted softplus
        # 5) atom-wise, 64
        V = self.out_atom_wise(X_conv)

        # Output: Residual
        X_residual = X + V

        return X_residual


class SchNetInteractionReg(nn.Module):
    """SchNet with additional regularization layers."""

    def __init__(
        self,
        atom_embeddings_dim: int,
        rbf_min: float,
        rbf_max: float,
        n_rbf: int,
        activation: nn.Module,
    ):
        super().__init__()
        self.in_atom_wise = nn.Linear(
            atom_embeddings_dim,
            atom_embeddings_dim,
            bias=False,
        )

        self.cf_conv = CfConvReg(
            atom_embeddings_dim, rbf_min, rbf_max, n_rbf, activation
        )

        self.out_atom_wise = nn.Sequential(
            nn.Linear(atom_embeddings_dim, atom_embeddings_dim, bias=False),
            RMSNorm(atom_embeddings_dim),
            activation(),
            nn.Linear(atom_embeddings_dim, atom_embeddings_dim),
        )

    def forward(
        self,
        X: torch.Tensor,
        R_distances: torch.Tensor,
        idx_i: torch.Tensor,
        idx_j: torch.Tensor,
    ):
        # 1) atom-wise, 64
        X_in = self.in_atom_wise(X)

        # 2) cfconv, 64
        X_conv = self.cf_conv(X_in, R_distances, idx_i, idx_j)

        # 3) atom-wise, 64
        # 4) shifted softplus
        # 5) atom-wise, 64
        V = self.out_atom_wise(X_conv)

        # Output: Residual
        X_residual = X + V

        return X_residual


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        """
        Root Mean Square Layer Normalization.

        From # https://github.com/bzhangGo/rmsnorm/blob/master/rmsnorm_torch.py

        Parameters
        ----------
        d : int
            model size
        p : float, optional
            partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        eps : float, optional
            epsilon value, default 1e-8
        bias : bool, optional
            Whether use bias term for RMSNorm, disabled by default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed
