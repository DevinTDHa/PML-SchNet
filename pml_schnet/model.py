from typing import Dict

import torch
from torch import nn

from pml_schnet.activation import ShiftedSoftPlus
from pml_schnet.layers import (
    SchNetInteraction,
    SchNetInteractionReg,
    RMSNorm,
)
from pml_schnet.loss import derive_force


class PairwiseDistances(nn.Module):
    """
    Compute pair-wise distances from indices provided by a neighbor list transform.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        R = inputs["R"]
        idx_i = inputs["idx_i"]
        idx_j = inputs["idx_j"]

        Rij = R[idx_i] - R[idx_j]
        d_ij = torch.norm(Rij, dim=-1)

        return d_ij

    @staticmethod
    def forward_sep(R, idx_i, idx_j) -> torch.Tensor:
        Rij = R[idx_i] - R[idx_j]
        d_ij = torch.norm(Rij, dim=-1)

        return d_ij


class BaselineModel(nn.Module):
    def __init__(self, max_atoms=100, embedding_dim=8, spatial_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(max_atoms, embedding_dim)
        self.spatial_processor = nn.Sequential(
            nn.Linear(3, spatial_dim), nn.ReLU(), nn.Linear(spatial_dim, spatial_dim)
        )
        self.combiner = nn.Linear(embedding_dim + spatial_dim, 1, bias=False)

    def forward(self, input):
        n_atoms = input["N"]
        Z = input["Z"]
        R = input["R"]
        embedded_Z = self.embedding(Z)
        processed_R = self.spatial_processor(R)
        combined_features = torch.cat((embedded_Z, processed_R), dim=1)
        y = self.combiner(combined_features)
        Y_batch = torch.split(y, n_atoms.tolist())
        # torch.stack keeps grad_fn
        batch_means = torch.stack([pred.sum() for pred in Y_batch])
        return batch_means


class SchNet(nn.Module):
    def __init__(
        self,
        atom_embedding_dim=64,
        n_interactions=3,
        max_z=100,
        rbf_min=0.0,
        rbf_max=30.0,
        n_rbf=300,
        activation: nn.Module = ShiftedSoftPlus,
        writer=None,
        running_mean_var=True,
    ):
        """
        # Molecular representation NOTES
        # Basically  convert atomic positions and nuclear charges (R,Z) into feature vector X.
        Nuclear charge can also be called "Atom-Type"

        # ATOMWISE NOTES:
        Atom wise layers SHARE WEIGHTS across atoms and have no activation func?. But what about layers?
        Atomwise are dense layers applied separately to representation of each atom in X.
        Layers recombine feature maps and share weights across atoms
        I guess we re-use all atom-wise with same shape and fetch them based on embedding


        # INTERACTION BLOCK NOTES
        """
        super().__init__()
        self.time_step = 0
        self.writer = writer
        self.max_z = max_z
        self.embedding = nn.Embedding(max_z, atom_embedding_dim, padding_idx=0)

        self.interactions = nn.ModuleList(
            [
                SchNetInteraction(
                    atom_embedding_dim, rbf_min, rbf_max, n_rbf, activation
                )
                for _ in range(n_interactions)
            ]
        )

        self.output_layers = nn.Sequential(
            nn.Linear(atom_embedding_dim, 32),
            activation(),
            nn.Linear(32, 1),
        )
        self.pairwise = PairwiseDistances()

        self.running_mean_var = running_mean_var
        if running_mean_var:
            from welford_torch import Welford

            self.welford_E = Welford()
            self.welford_F = Welford()

    def forward(self, inputs: Dict):
        Z = inputs["Z"]  # Atomic numbers

        N = inputs["N"]  # Number of atoms in each molecule

        R_distances = self.pairwise(inputs)

        idx_i = inputs["idx_i"]  # indices of center atoms
        idx_j = inputs["idx_j"]  # indices of neighboring atoms

        # 1) Embedding 64 (See section Molecular representation).
        X = self.embedding(Z)

        # 2),3),4) each Interaction 64 with residual layers
        X_interacted = X
        for i, interaction in enumerate(self.interactions):
            X_interacted = interaction(X_interacted, R_distances, idx_i, idx_j)
            # if self.writer:
            #     self.writer.add_histogram(
            #         f"interaction_{i}", X_interacted, self.time_step
            #     )
            # self.writer.add_scalar(f'Gradient_norm/{name}', param.grad.norm(), epoch)

        # 5) atom-wise 32
        # 6) Shifted Softplus
        # 7) atom-wise 1
        atom_outputs = self.output_layers(X_interacted)
        # if self.writer:
        #     self.writer.add_histogram(f"atom_outputs", atom_outputs, self.time_step)

        # Assign Flattened Atoms Back to Molecules
        atom_partitions = torch.split(
            atom_outputs, N.tolist() if isinstance(N, torch.Tensor) else N
        )

        # 8) Sum Pooling
        predicted_energies = torch.stack([p.sum() for p in atom_partitions])
        self.time_step += 1
        return predicted_energies

    def forward_sep(self, embed_Z, R, N, idx_i, idx_j):
        R_distances = self.pairwise.forward_sep(R, idx_i, idx_j)

        # 1) Embedding 64 (See section Molecular representation).
        X = embed_Z

        # 2),3),4) each Interaction 64 with residual layers
        X_interacted = X
        for i, interaction in enumerate(self.interactions):
            X_interacted = interaction(X_interacted, R_distances, idx_i, idx_j)

        # 5) atom-wise 32
        # 6) Shifted Softplus
        # 7) atom-wise 1
        atom_outputs = self.output_layers(X_interacted)
        # if self.writer:
        #     self.writer.add_histogram(f"atom_outputs", atom_outputs, self.time_step)

        # Assign Flattened Atoms Back to Molecules
        atom_partitions = torch.split(
            atom_outputs, N.tolist() if isinstance(N, torch.Tensor) else N
        )

        # 8) Sum Pooling
        predicted_energies = torch.stack([p.sum() for p in atom_partitions])
        self.time_step += 1
        return predicted_energies

    def update_mean_var(self, E, F):
        self.welford_E.add_all(E.view(-1, 1))
        self.welford_F.add_all(F)

    def get_mean(self):
        if self.running_mean_var:
            return self.welford_E.mean, self.welford_F.mean
        else:
            raise ValueError("Mean and Var are not tracked.")

    def get_var(self):
        if self.running_mean_var:
            return self.welford_E.var_s, self.welford_F.var_s
        else:
            raise ValueError("Mean and Var are not tracked.")

    def get_std(self):
        if self.running_mean_var:
            return torch.sqrt(self.welford_E.var_s), torch.sqrt(self.welford_F.var_s)
        else:
            raise ValueError("Mean and Var are not tracked.")

    def predict(self, x):
        """
        Perform prediction on the input data and calculate z-scores for energy and force predictions for a
        confidence measure.

        Parameters
        ----------
        x : dict
            Input data dictionary for inference

        Returns
        -------
        Tuple
            Predicted energy and force, and z-scores for energy and force
        """
        x["R"].requires_grad_()
        pred_E = self.forward(x)
        pred_F = derive_force(pred_E, x["R"])

        mean_E, mean_F = self.get_mean()
        var_E, var_F = self.get_var()

        batch_size = len(x["N"])
        z_score_E = (mean_E - pred_E).abs() / torch.sqrt(var_E)
        z_score_F = (mean_F - pred_F).abs() / torch.sqrt(var_F)
        z_score_F = z_score_F.view(batch_size, -1, 3)

        return (pred_E, pred_F), (z_score_E, z_score_F)

    def explain(self, x):
        """
        Explain the given input x. We use the sensitivity as a measure of feature importance. The atom positions R and
        the embeddings of the atom numbers Z are explained.

        Parameters
        ----------
        x : dict
            A dictionary containing keys "Z", "R", "N", "idx_i", and "idx_j".

        Returns
        -------
        tuple
            A tuple containing the sensitivity of embedding Z and R.
        """
        Z = x["Z"]
        R = x["R"]
        N = x["N"]
        idx_i = x["idx_i"]
        idx_j = x["idx_j"]

        embed_Z = self.embedding(Z)
        embed_Z.requires_grad_()
        R.requires_grad_()

        embed_Z = self.embedding(Z)
        output = self.forward_sep(embed_Z, R, N, idx_i, idx_j)

        embed_Z_grad, R_grad = torch.autograd.grad(output, [embed_Z, R])
        embed_Z_sensitivity, R_sensitivity = embed_Z_grad.pow(2).norm(
            dim=1
        ), R_grad.pow(2)
        return embed_Z_sensitivity, R_sensitivity


class SchNetRMSNorm(SchNet):
    def __init__(
        self,
        atom_embedding_dim=64,
        n_interactions=3,
        max_z=100,
        rbf_min=0.0,
        rbf_max=30.0,
        n_rbf=300,
        activation: nn.Module = ShiftedSoftPlus,
        running_mean_var=True,
    ):
        super().__init__()
        self.time_step = 0
        # self.writer = writer
        self.max_z = max_z
        self.embedding = nn.Embedding(max_z, atom_embedding_dim, padding_idx=0)

        self.interactions = nn.ModuleList(
            [
                SchNetInteractionReg(
                    atom_embedding_dim, rbf_min, rbf_max, n_rbf, activation
                )
                for _ in range(n_interactions)
            ]
        )

        self.output_layers = nn.Sequential(
            nn.Linear(atom_embedding_dim, 32, bias=False),
            RMSNorm(32),
            activation(),
            nn.Linear(32, 1),
        )
        self.pairwise = PairwiseDistances()

        self.running_mean_var = running_mean_var
        if running_mean_var:
            from welford_torch import Welford

            self.welford_E = Welford()
            self.welford_F = Welford()


class SchNetDropout(SchNet):
    def forward(self, inputs: Dict):
        Z = inputs["Z"]  # Atomic numbers

        N = inputs["N"]  # Number of atoms in each molecule

        R_distances = self.pairwise(inputs)

        idx_i = inputs["idx_i"]  # indices of center atoms
        idx_j = inputs["idx_j"]  # indices of neighboring atoms

        # 1) Embedding 64 (See section Molecular representation).
        X = self.embedding(Z)

        dropout = nn.Dropout()

        # 2),3),4) each Interaction 64 with residual layers
        X_interacted = X
        for i, interaction in enumerate(self.interactions):
            X_interacted = interaction(X_interacted, R_distances, idx_i, idx_j)
            if i == 0:
                X_interacted = dropout(X_interacted)

        # 5) atom-wise 32
        # 6) Shifted Softplus
        # 7) atom-wise 1
        atom_outputs = self.output_layers(X_interacted)
        # if self.writer:
        #     self.writer.add_histogram(f"atom_outputs", atom_outputs, self.time_step)

        # Assign Flattened Atoms Back to Molecules
        atom_partitions = torch.split(
            atom_outputs, N.tolist() if isinstance(N, torch.Tensor) else N
        )

        # 8) Sum Pooling
        predicted_energies = torch.stack([p.sum() for p in atom_partitions])
        self.time_step += 1
        return predicted_energies


class SchNetTransformer(nn.Module):
    def __init__(
        self,
        max_z=100,
        atom_embedding_dim=64,
        d_model=64,
        nhead=8,
        dim_feedforward=64,
        dropout=0,
        transformer_layers_Z=3,
        transformer_layers_R=2,
        max_atom_length=19,
        activation: nn.Module = ShiftedSoftPlus,
        running_mean_var=False,
    ):
        super().__init__()
        self.time_step = 0
        self.max_z = max_z
        self.embedding = nn.Embedding(max_z, atom_embedding_dim, padding_idx=0)

        self.transformer_Z = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward, dropout, batch_first=True
            ),
            num_layers=transformer_layers_Z,
        )

        self.transformer_R = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                max_atom_length * 3, 3, dim_feedforward, dropout, batch_first=True
            ),
            num_layers=transformer_layers_R,
        )

        self.output_Z = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(max_atom_length * d_model, 32),
            activation(),
        )

        self.output_R = nn.Sequential(nn.Linear(max_atom_length * 3, 32), activation())

        self.linear_out = nn.Linear(32, 1)

        self.running_mean_var = running_mean_var
        if running_mean_var:
            from welford_torch import Welford

            self.welford_E = Welford()
            self.welford_F = Welford()

    def forward(self, inputs: Dict):
        Z = inputs["Z"]  # Atomic numbers
        N = inputs["N"]  # Number of atoms in each molecule
        R = inputs["R"]
        batch_size = len(N)

        stacked_Z = torch.stack(torch.split(Z, N))

        embed_stacked_Z = self.embedding(stacked_Z)

        transformer_out_Z = self.transformer_Z(embed_stacked_Z)
        out_Z = self.output_Z(transformer_out_Z)
        transformer_out_R = self.transformer_R(R.view(batch_size, -1))
        out_R = self.output_R(transformer_out_R)

        combined_out = out_Z + out_R

        predicted_energies = self.linear_out(combined_out)

        self.time_step += 1
        return predicted_energies.flatten()
