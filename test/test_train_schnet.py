import numpy as np
import pytest
import schnetpack.properties as structure
import torch
from ase.neighborlist import neighbor_list

from pml_schnet.model import SchnetNet
from pml_schnet.route import train_and_validate
from pml_schnet.settings import Dataset, Model
from pml_schnet.settings import (
    Trainable,
)
from ase import Atoms

@pytest.fixture(scope="session")
def num_data():
    return 20

@pytest.fixture(scope="session")
def example_data(num_data):
    data = []
    for i in range(1, num_data + 1):

        mol_1 = torch.normal(1.0, 0.01, size=(3, 3))
        z = np.random.randint(1, 100, size=(len(mol_1),))
        ats = Atoms(numbers=z, positions=mol_1.numpy(), cell=None, pbc=False)
        data.append(ats)
    return data


@pytest.fixture
def indexed_data(example_data):
    # batch_size =5
    batch_size = 3
    Z = []
    R = []
    C = []
    seg_m = []
    ind_i = []
    ind_j = []
    ind_S = []
    Rij = []

    n_atoms = 0
    n_pairs = 0
    quantities: str
    # Quantities to compute by the neighbor list algorithm. Each character
    # in this string defines a quantity. They are returned in a tuple of
    # the same order. Possible quantities are:
    #
    # * 'i' : first atom index
    # * 'j' : second atom index
    # * 'd' : absolute distance
    # * 'D' : distance vector
    # * 'S' : shift vector (number of cell boundaries crossed by the bond
    # between atom i and j). With the shift vector S, the
    # distances D between atoms can be computed from:
    # D = a.positions[j]-a.positions[i]+S.dot(a.cell)
    for i in range(len(example_data)):
        seg_m.append(n_atoms)
        atoms = example_data[i]
        atoms.set_pbc(False)
        Z.append(atoms.numbers)
        R.append(atoms.positions)
        C.append(atoms.cell)
        idx_i, idx_j, idx_S, rij = neighbor_list(
            "ijSD", atoms, np.inf, self_interaction=False
        )
        _, seg_im = np.unique(idx_i, return_counts=True)
        ind_i.append(idx_i + n_atoms)
        ind_j.append(idx_j + n_atoms)
        ind_S.append(idx_S)
        Rij.append(rij.astype(np.float32))
        n_atoms += len(atoms)
        n_pairs += len(idx_i)
        if i + 1 >= batch_size:
            break
    seg_m.append(n_atoms)

    Z = np.hstack(Z)
    R = np.vstack(R).astype(np.float32)
    C = np.array(C).astype(np.float32)
    seg_m = np.hstack(seg_m)
    ind_i = np.hstack(ind_i)
    ind_j = np.hstack(ind_j)
    ind_S = np.vstack(ind_S)
    Rij = np.vstack(Rij)

    inputs = {
        structure.Z: torch.tensor(Z),
        structure.position: torch.tensor(R),
        structure.idx_m: torch.tensor(seg_m),
        structure.idx_j: torch.tensor(ind_j),
        structure.idx_i: torch.tensor(ind_i),
        structure.Rij: torch.tensor(Rij),
        # structure.cell: torch.tensor(C),
    }

    return inputs


# all_trainable
def test_id(test: Trainable) -> str:
    return f"{test.dataset}_{test.task}_{test.molecule}"


def test_data_gen(indexed_data):
    print(indexed_data)


# @pytest.mark.parametrize("trainable", iso17_trainable, ids=test_id)
def test_train_iso17_energy_force(trainable: Trainable):

    model = SchnetNet(n_atom_basis=128, n_interactions=3, radial_basis=, cutoff_fn=, max_z=100)
    train_loss, test_loss = train_and_validate(
        trainable, Model.schnet, n_train=200000, n_test=100
    )
