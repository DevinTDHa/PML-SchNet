import torch
from schnetpack.nn import Dense
from torch import nn

from pml_schnet.activation import ShiftedSoftPlus


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
    def __init__(self, n_atom_basis, n_filters, radial_basis, cutoff_fn, max_z=100):
        super().__init__()
        self.n_atom_basis = n_atom_basis
        self.size = (self.n_atom_basis,)
        self.n_filters = n_filters or self.n_atom_basis
        self.radial_basis = radial_basis
        self.cutoff_fn = cutoff_fn
        self.cutoff = cutoff_fn.cutoff
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

        # 1) Embedding 64 ( see section Molecular representation).
        self.embedding = nn.Embedding(max_z, self.n_atom_basis, padding_idx=0)

        # 2) 3x Interaction 64
        # TODO Interaction block implementation

        # 3) Atomwise 32. No Bias/Activation
        self.atom_wise_32 = Dense(64, 32, bias=False, activation=None)
        # 4) Shifted Softplus
        self.shifted_softplus = ShiftedSoftPlus()

        # 5) atom-wise 1 # Todo bias/activation?
        self.atom_wise_1 = Dense(32, 1, bias=True, activation=None)

        # 6) Sum pooling

        # 7) E_hat

    def forward(self, input):

        # TODO 2) Interaction blocks are applied residually
        # i.e. x = x + v
        raise NotImplementedError("todo")
