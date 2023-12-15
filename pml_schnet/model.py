from typing import Dict

import torch
from torch import nn

from pml_schnet.activation import ShiftedSoftPlus
from pml_schnet.layers import SchNetInteraction


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

        Rij = R[idx_j] - R[idx_i]  # + offsets # TODO?
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


class SchnetNet(nn.Module):
    def __init__(
        self,
        atom_embedding_dim=64,
        n_interactions=3,
        max_z=100,
        rbf_min=0.0,
        rbf_max=30.0,
        n_rbf=300,
        activation: nn.Module = ShiftedSoftPlus,
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
         t
        """
        super().__init__()

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
            nn.Linear(atom_embedding_dim, 32),  # , bias=False),  # TODO: why no bias?
            activation(),
            nn.Linear(32, 1),
        )
        self.pairwise = PairwiseDistances()

    def forward(self, inputs: Dict):
        Z = inputs["Z"]  # Atomic numbers. Label

        N = inputs["N"]  # Number of atoms in each molecule
        # AKA R_ij/  vectors pointing from center atoms to neighboring atoms
        # R_distances = inputs["d"]

        R_distances = self.pairwise(inputs)

        idx_i = inputs["idx_i"]  # indices of center atoms
        idx_j = inputs["idx_j"]  # indices of neighboring atoms

        # 1) Embedding 64 ( see section Molecular representation).
        X = self.embedding(Z)

        # 2),3),4) each Interaction 64 with recurrent layers
        X_interacted = X
        for interaction in self.interactions:
            X_interacted = interaction(X_interacted, R_distances, idx_i, idx_j)

        # 5) atom-wise 32
        # 6) Shifted Softplus
        # 7) atom-wise 1
        atom_outputs = self.output_layers(X_interacted)

        # Assign Flattened Atoms Back to Molecules
        atom_partitions = torch.split(atom_outputs, N)

        # 8) Sum Pooling
        predicted_energies = torch.stack([p.sum() for p in atom_partitions])

        return predicted_energies
