import torch
from torch import nn


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
