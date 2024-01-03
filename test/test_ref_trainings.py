import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from pml_schnet.data_loader import load_data
from pml_schnet.loss import energy_force_loss, derive_force
from pml_schnet.model import SchNet
from pml_schnet.settings import (
    device,
)
from training import validate_schnet_force_energy, validate_schnet_energy


def save_model(model, path):
    torch.save(model, path + ".pt")


def test_ref_train_iso17_long():
    """10000 train data points, 1000 test data points, in mini-batches of size 32"""
    model = SchNet().to(device)

    total_data = 404000
    n_train = int(total_data * 0.8)
    n_test = int(total_data * 0.2)
    batch_size = 64

    epochs = 2
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    with tqdm(total=epochs, ncols=80) as progress_bar:
        progress_bar.set_description("Schnet ISO17")
        for epoch in range(epochs):
            train_gen, test_gen = load_data(
                "ISO17",
                n_train,
                n_test,
                batch_size=batch_size,
                split_file="test_ref_train_iso17_long.split",
            )
            if epoch == epochs - 1:
                print("Last epoch, setting LR to 0.001")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

                progress_bar.set_postfix(train_loss=f"{loss:.4E}")
                losses.append(loss.item())

            # Validation Losses
            val_losses = validate_schnet_force_energy(model, test_gen, nn.L1Loss())
            print(f"Validation Loss Epoch {epoch}", np.mean(val_losses))
            progress_bar.update(1)

    np.save("iso17_long_train_losses.npy", arr=losses)
    np.save("iso17_long_val_losses.npy", arr=val_losses)
    save_model(model, "schnet_iso17")


def test_ref_train_QM9():
    """10000 train data points, 1000 test data points, in mini-batches of size 32"""
    model = SchNet().to(device)

    n_train = 50000
    n_test = int(n_train * 0.2)
    batch_size = 64

    epochs = 2
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    criterion = nn.L1Loss()
    with tqdm(total=epochs, ncols=80) as progress_bar:
        progress_bar.set_description("Schnet QM9")
        for epoch in range(epochs):
            train_gen, test_gen = load_data(
                "QM9",
                n_train,
                n_test,
                batch_size=batch_size,
                split_file="test_ref_train_QM9.split",
            )
            if epoch == epochs - 1:
                print("Last epoch, setting LR to 0.001")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for X_batch, y_batch in train_gen:
                # Forward pass
                pred = model(X_batch)
                loss = criterion(pred, y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                progress_bar.set_postfix(train_loss=f"{loss:.4E}")
                losses.append(loss.item())

            # Validation Losses
            val_losses = validate_schnet_energy(model, test_gen, criterion)
            print(f"Validation Loss Epoch {epoch}", np.mean(val_losses))
            progress_bar.update(1)

    np.save("qm9_train_losses.npy", arr=losses)
    np.save("qm9_val_losses.npy", arr=val_losses)
    save_model(model, "schnet_qm9")


def test_ref_train_md17_aspirin_energy():
    model = SchNet().to(device)

    n_train = 50000
    n_test = int(n_train * 0.2)
    batch_size = 64

    epochs = 2
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    criterion = nn.L1Loss()
    with tqdm(total=epochs, ncols=80) as progress_bar:
        progress_bar.set_description("Schnet MD17: Energy")
        for epoch in range(epochs):
            train_gen, test_gen = load_data(
                "MD17",
                n_train,
                n_test,
                batch_size=batch_size,
                split_file="ref_train_MD17.split",
            )
            if epoch == epochs - 1:
                print("Last epoch, setting LR to 0.001")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for X_batch, y_batch in train_gen:
                # Forward pass
                pred = model(X_batch)
                loss = criterion(pred, y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                progress_bar.set_postfix(train_loss=f"{loss:.4E}")
                losses.append(loss.item())

            # Validation Losses
            val_losses = validate_schnet_energy(model, test_gen, criterion)
            print(f"Validation Loss Epoch {epoch}", np.mean(val_losses))
            progress_bar.update(1)

    np.save("md17_energy_train_losses.npy", arr=losses)
    np.save("md17_energy_val_losses.npy", arr=val_losses)
    save_model(model, "schnet_md17_energy")


def test_ref_train_md17_aspirin_force():
    model = SchNet().to(device)

    n_train = 50000
    n_test = int(n_train * 0.2)
    batch_size = 64

    epochs = 2
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    criterion = nn.L1Loss()
    with tqdm(total=epochs, ncols=80) as progress_bar:
        progress_bar.set_description("Schnet MD17: Force")
        for epoch in range(epochs):
            train_gen, test_gen = load_data(
                "MD17",
                n_train,
                n_test,
                batch_size=batch_size,
                split_file="ref_train_MD17.split",
            )
            if epoch == epochs - 1:
                print("Last epoch, setting LR to 0.001")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            for X_batch, y_batch in train_gen:
                # Forward pass
                X_batch["N"] = X_batch["N"]
                X_batch["Z"] = X_batch["Z"]
                X_batch["R"] = X_batch["R"]
                X_batch["R"].requires_grad_()

                target_F = X_batch["F"].to(device)
                target_F.requires_grad_()

                E_pred = model(X_batch)

                F_pred = derive_force(E_pred, X_batch["R"])

                loss = criterion(F_pred, target_F)

                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                progress_bar.set_postfix(train_loss=f"{loss:.4E}")
                losses.append(loss.item())

            # Validation Losses
            val_losses = validate_schnet_energy(model, test_gen, criterion)
            print(f"Validation Loss Epoch {epoch}", np.mean(val_losses))
            progress_bar.update(1)

    np.save("md17_force_train_losses.npy", arr=losses)
    np.save("md17_force_val_losses.npy", arr=val_losses)
    save_model(model, "schnet_md17_force")


def test_ref_train_md17_aspirin_energy_force():
    model = SchNet().to(device)

    n_train = 50000
    n_test = int(n_train * 0.2)
    batch_size = 64

    epochs = 2
    lr = 0.01
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    with tqdm(total=epochs, ncols=80) as progress_bar:
        progress_bar.set_description("Schnet MD17: Energy Force")
        for epoch in range(epochs):
            train_gen, test_gen = load_data(
                "MD17",
                n_train,
                n_test,
                batch_size=batch_size,
                split_file="ref_train_MD17.split",
            )
            if epoch == epochs - 1:
                print("Last epoch, setting LR to 0.001")
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
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

                progress_bar.set_postfix(train_loss=f"{loss:.4E}")
                losses.append(loss.item())

            # Validation Losses
            val_losses = validate_schnet_force_energy(model, test_gen, nn.L1Loss())
            print(f"Validation Loss Epoch {epoch}", np.mean(val_losses))
            progress_bar.update(1)

    np.save("md17_energy_force_val_losses.npy", arr=val_losses)
    np.save("md17_energy_force_train_losses.npy", arr=losses)
    save_model(model, "schnet_md17_energy_force")
