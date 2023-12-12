import torch
from torch import nn
from tqdm import tqdm

import pml_schnet.route
from pml_schnet.settings import device
from pml_schnet.visualization.plotting import plot_loss
from pml_schnet.data_loader import load_data
from pml_schnet.loss import derive_force, energy_force_loss


def train_baseline_energy(model, n_train, n_test, lr, epochs, dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    losses = []
    for epoch in tqdm(range(epochs)):
        train_gen, test_gen = load_data(dataset, n_train, n_test)
        loss = None
        for X_batch, y_batch in train_gen:
            # Forward pass
            X_batch["N"] = X_batch["N"].to(device)
            X_batch["Z"] = X_batch["Z"].to(device)
            X_batch["R"] = X_batch["R"].to(device)
            y_batch = y_batch.to(device)
            loss = criterion(model(X_batch), y_batch)
            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
        print(f"Epoch {epoch + 1}, Train Loss: {loss:.4f}")
        losses.append({"epoch": epoch, "loss": loss.item()})
    # plot_loss(losses)
    return losses[-1]["loss"]


def train_baseline_force(model, n_train, n_test, lr, epochs, dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    losses = []
    for epoch in tqdm(range(epochs)):
        train_gen, test_gen = load_data(dataset, n_train, n_test)
        loss = None
        for X_batch, y_batch in train_gen:
            # Forward pass
            X_batch["N"] = X_batch["N"].to(device)
            X_batch["Z"] = X_batch["Z"].to(device)
            X_batch["R"] = X_batch["R"].to(device)
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
        print(f"Epoch {epoch + 1}, Train Loss: {loss:.4f}")
        losses.append({"epoch": epoch, "loss": loss.item()})
    plot_loss(losses)
    return losses[-1]["loss"]


def train_baseline_energy_force(model, n_train, n_test, lr, epochs, dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []

    for epoch in tqdm(range(epochs)):
        train_gen, test_gen = load_data(dataset, n_train, n_test)
        loss = None
        for X_batch, y_batch in train_gen:
            # Forward pass
            X_batch["N"] = X_batch["N"].to(device)
            X_batch["Z"] = X_batch["Z"].to(device)
            X_batch["R"] = X_batch["R"].to(device)

            X_batch["R"].requires_grad_()
            F = X_batch["F"].to(device)
            y_batch = y_batch.to(device)
            E_pred = model(X_batch)
            loss = energy_force_loss(E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F)
            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
        print(f"Epoch {epoch + 1}, Train Loss: {loss:.4f}")
        losses.append({"epoch": epoch, "loss": loss.item()})
    plot_loss(losses)
    return losses[-1]["loss"]



def train_schnet_energy(
        model_obj, n_train, n_test, lr, epochs, dataset
) : pass


def train_schnet_energy_force(
        model_obj, n_train, n_test, lr, epochs, dataset
) : pass


def train_schnet_force(
        model_obj, n_train, n_test, lr, epochs, dataset
) : pass
