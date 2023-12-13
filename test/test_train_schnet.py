import numpy as np
import pytest
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch import nn, autograd
from torch.nn import ELU
from tqdm import tqdm

from data_loader import load_data
from loss import energy_force_loss
from pml_schnet.model import SchnetNet
from pml_schnet.settings import (
    Trainable,
    device,
)


@pytest.fixture(scope="session")
def num_data():
    return 2


@pytest.fixture(scope="session")
def example_data(num_data):
    data = []
    for i in range(1, num_data + 1):
        if i == 1:
            n_atoms = 3
            r = np.random.normal(1, 0.01, (n_atoms, 3))
        elif i == 2:
            n_atoms = 4
            r = np.random.normal(-100, 10, (n_atoms, 3))
        else:
            n_atoms = 0
            r = np.random.randn(n_atoms, 3)

        z = np.random.randint(1, 100, size=(len(r),))
        ats = Atoms(numbers=z, positions=r, cell=None, pbc=False)
        data.append(ats)
    return data


@pytest.fixture
def indexed_data(example_data):
    # batch_size =5
    batch_size = len(example_data)
    Z = []
    N = []
    R = []
    C = []
    # seg_m = []
    ind_i = []
    ind_j = []
    d = []

    n_atoms = 0
    n_pairs = 0

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
        # seg_m.append(n_atoms)
        atoms = example_data[i]
        atoms.set_pbc(False)
        Z.append(atoms.numbers)
        N.append(len(atoms.numbers))
        R.append(atoms.positions)
        C.append(atoms.cell)
        idx_i, idx_j, dij = neighbor_list("ijd", atoms, np.inf, self_interaction=False)
        _, seg_im = np.unique(idx_i, return_counts=True)
        ind_i.append(idx_i + n_atoms)
        ind_j.append(idx_j + n_atoms)
        d.append(dij.astype(np.float32))
        n_atoms += len(atoms)
        n_pairs += len(idx_i)
        if i + 1 >= batch_size:
            break
    # seg_m.append(n_atoms)

    Z = np.hstack(Z)
    R = np.vstack(R).astype(np.float32)
    ind_i = np.hstack(ind_i)
    ind_j = np.hstack(ind_j)
    d = np.hstack(d)

    inputs = {
        "Z": torch.tensor(Z),
        "N": N,
        "R": torch.tensor(R),
        "idx_j": torch.tensor(ind_j),
        "idx_i": torch.tensor(ind_i),
        "d": torch.tensor(d),
    }

    return inputs


# all_trainable
def test_id(test: Trainable) -> str:
    return f"{test.dataset}_{test.task}_{test.molecule}"


def test_data_gen(indexed_data):
    print(indexed_data)


def test_inference(indexed_data):
    model = SchnetNet(activation=ELU)

    res = model(indexed_data)
    print(res)


def test_train_schnet():
    model = SchnetNet(activation=ELU).to(device)
    lr = 0.1

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    losses = []

    epochs = 5
    with autograd.detect_anomaly():
        for epoch in tqdm(range(epochs)):
            train_gen, test_gen = load_data("QM9", 10000, 10, batch_size=128)
            loss = None
            for X_batch, y_batch in train_gen:
                # Forward pass
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights
            print(f"Epoch {epoch + 1}, Train Loss: {loss:.4f}")
            losses.append({"epoch": epoch, "loss": loss.item()})
    return losses[-1]["loss"]


def test_train_schnet_force():
    model = SchnetNet(activation=ELU).to(device)
    lr = 0.1

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    epochs = 5
    with autograd.detect_anomaly():
        for epoch in tqdm(range(epochs)):
            train_gen, test_gen = load_data("MD17", 10000, 10, batch_size=128)
            loss = None
            for X_batch, y_batch in train_gen:
                # Forward pass
                X_batch["R"].requires_grad_()
                F = X_batch["F"].to(device)

                # Forward pass
                E_pred = model(X_batch)
                loss = energy_force_loss(E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F)
                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights
            print(f"Epoch {epoch + 1}, Train Loss: {loss:.4f}")
            losses.append({"epoch": epoch, "loss": loss.item()})
    return losses[-1]["loss"]
