import numpy as np
import pytest
import torch
from ase import Atoms
from ase.neighborlist import neighbor_list
from torch import nn, autograd
from tqdm import tqdm

from pml_schnet.data_loader import load_data
from pml_schnet.loss import energy_force_loss, derive_force
from pml_schnet.model import SchNet
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


def test_inference_dummy_data(indexed_data):
    model = SchNet()

    res = model(indexed_data)
    print(res)


def test_train_schnet_energy():
    model = SchNet().to(device)
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    losses = []

    epochs = 1000
    train_gen, test_gen = load_data("QM9", 8, 10, batch_size=8)
    train_gen, test_gen = list(train_gen), list(test_gen)

    with autograd.detect_anomaly():
        with tqdm(total=epochs, ncols=80) as progress_bar:
            progress_bar.set_description("Schnet E")
            for epoch in range(epochs):
                loss = None
                for X_batch, y_batch in train_gen:
                    # Forward pass
                    pred = model(X_batch)
                    loss = criterion(pred, y_batch)
                    # Backward pass and optimization
                    optimizer.zero_grad()  # Clear gradients
                    loss.backward()  # Compute gradients
                    optimizer.step()  # Update weights

                    progress_bar.set_postfix(train_loss=f"{loss:.4E}")
                progress_bar.update(1)
                losses.append(loss.item())

    print([f"{l:.4E}" for l in losses])

    validate_test(model, test_gen, criterion)


def validate_test(model, test_gen, criterion):
    val_loss = []
    for X_batch, y_batch in test_gen:
        # Forward pass
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        val_loss.append(loss.item())

    print(val_loss, np.mean(val_loss))


def test_train_schnet_force_and_energy():
    model = SchNet().to(device)
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    epochs = 1000
    with autograd.detect_anomaly():
        with tqdm(total=epochs, ncols=80) as progress_bar:
            progress_bar.set_description("Schnet E+F")
            for epoch in range(epochs):
                train_gen, test_gen = load_data("MD17", 8, 64, batch_size=8)
                loss = None
                for X_batch, y_batch in train_gen:
                    # Forward pass
                    X_batch["R"].requires_grad_()
                    F = X_batch["F"].to(device)

                    # Forward pass
                    E_pred = model(X_batch)
                    loss = energy_force_loss(
                        E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F
                    )
                    # Backward pass and optimization
                    optimizer.zero_grad()  # Clear gradients
                    loss.backward()  # Compute gradients

                    print_mean_grad_stats(model)
                    optimizer.step()  # Update weights

                    progress_bar.set_postfix(train_loss=f"{loss:.4E}")
                progress_bar.update(1)
                losses.append(loss.item())
    print([f"{l:.4E}" for l in losses])


def test_train_schnet_force():
    model = SchNet().to(device)
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    criterion = nn.L1Loss()

    epochs = 1000
    train_gen, test_gen = load_data("QM9", 8, 10, batch_size=8)
    train_gen, test_gen = list(train_gen), list(test_gen)

    with autograd.detect_anomaly():
        with tqdm(total=epochs, ncols=80) as progress_bar:
            for epoch in range(epochs):
                train_gen, test_gen = load_data("MD17", 64, 64, batch_size=64)
                loss = None
                for X_batch, y_batch in train_gen:
                    # Forward pass
                    X_batch["R"].requires_grad_()
                    E_pred = model(X_batch)
                    F_pred = derive_force(E_pred, X_batch["R"])
                    F_pred.requires_grad_()

                    target_F = X_batch["F"].to(device)
                    loss = criterion(F_pred, target_F)
                    # Backward pass and optimization
                    optimizer.zero_grad()  # Clear gradients
                    loss.backward()  # Compute gradients
                    optimizer.step()  # Update weights

                progress_bar.set_description("Schnet F")
                progress_bar.set_postfix(train_loss=f"{loss:.4E}")
                progress_bar.update(1)
                losses.append(loss.item())
    print([f"{l:.4E}" for l in losses])

    validate_test(model, test_gen, criterion)


def print_grad_stats(model):
    parameters_w_grads = [(pname, p.grad) for pname, p in model.named_parameters()]

    for pname, pgrad in parameters_w_grads:
        formatted_str = f"{pname:40s}: "
        for stat in [
            pgrad.min().item(),
            pgrad.max().item(),
            pgrad.mean().item(),
            pgrad.std().item(),
        ]:
            formatted_str += f"{stat:.2E}\t"
        print(formatted_str)


def print_mean_grad_stats(model: SchNet):
    parameter_grad_mins = np.array([p.grad.min().item() for p in model.parameters()])
    parameter_grad_maxs = np.array([p.grad.max().item() for p in model.parameters()])
    grad_avg_mins = parameter_grad_mins.mean()
    grad_avg_maxs = parameter_grad_maxs.mean()

    print(f"Avg. Min: {grad_avg_mins:.2E}, Avg. Max : {grad_avg_maxs:.2E}")


def test_train_schnet_force_iso17():
    model = SchNet().to(device)
    lr = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    criterion = nn.L1Loss()

    epochs = 100
    train_gen, test_gen = load_data("ISO17", 8, 128, batch_size=8)
    train_gen, test_gen = list(train_gen), list(test_gen)

    with autograd.detect_anomaly():
        with tqdm(total=epochs, ncols=80) as progress_bar:
            for epoch in range(epochs):
                loss = None
                for X_batch, y_batch in train_gen:
                    # Forward pass
                    X_batch["R"].requires_grad_()
                    E_pred = model(X_batch)
                    F_pred = derive_force(E_pred, X_batch["R"])
                    F_pred.requires_grad_()

                    target_F = X_batch["F"].to(device)
                    loss = criterion(F_pred, target_F)
                    # Backward pass and optimization
                    optimizer.zero_grad()  # Clear gradients
                    loss.backward()  # Compute gradients
                    optimizer.step()  # Update weights

                progress_bar.set_description("Schnet F")
                progress_bar.set_postfix(train_loss=f"{loss:.4E}")
                progress_bar.update(1)
                losses.append(loss.item())
    print([f"{l:.4E}" for l in losses])

    validate_test(model, test_gen, criterion)
