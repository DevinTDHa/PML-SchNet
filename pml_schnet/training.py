import os

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from pml_schnet.data_loader import load_data
from pml_schnet.loss import derive_force, energy_force_loss
from pml_schnet.settings import device
from pml_schnet.visualization.plotting import plot_loss


def write_grads(model, writer, epoch):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # print(f'name={name}, shape={param.grad.shape}')
            writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
            writer.add_scalar(f"Gradient_norm/{name}", param.grad.norm(), epoch)


def validate_schnet(model, test_gen, criterion):
    val_loss = []
    labels = []
    for X_batch, y_batch in test_gen:
        # Forward pass
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        labels.append(pred.item())
        val_loss.append(loss.item())

    return np.mean(val_loss),labels


def validate_schnet_force_energy(model, test_gen):
    val_loss = []
    labels = []
    for X_batch, y_batch in test_gen:
        # Forward pass
        X_batch["R"].requires_grad_()
        F = X_batch["F"].to(device)

        # Forward pass
        E_pred = model(X_batch)
        loss = energy_force_loss(E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F)
        labels.append(derive_force(E_pred, X_batch["R"]).item())

        val_loss.append(loss)

    return val_loss[0].item(),labels


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


def train_schnet_energy(model_obj, n_train, n_test, lr, epochs, dataset, batch_size):
    writer = model_obj.writer
    optimizer = torch.optim.Adam(model_obj.parameters(), lr=lr)
    criterion = nn.L1Loss()
    losses = []
    with tqdm(total=epochs, ncols=80) as progress_bar:
        for epoch in range(epochs):
            train_gen, test_gen = load_data(
                dataset, n_train, n_test, batch_size=batch_size
            )
            loss = None
            for X_batch, y_batch in train_gen:
                # Forward pass
                pred = model_obj(X_batch)
                loss = criterion(pred, y_batch)
                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

            progress_bar.set_description("Schnet E")
            progress_bar.set_postfix(train_loss=f"{loss:.4E}")
            progress_bar.update(1)
            losses.append(loss.item())
            writer.add_scalar("Loss", loss.item(), epoch)
            writer.add_scalar(
                "Validation", validate_schnet(model_obj, test_gen, criterion), epoch
            )
            write_grads(model_obj, writer, epoch)
            checkpoint_path = os.path.join(writer.log_dir, f"model_epoch_{epoch}.ckpt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_obj.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                checkpoint_path,
            )
    X_batch["N"] = torch.tensor(X_batch["N"])
    writer.add_graph(model_obj, (X_batch))
    return losses[-1]

def train_schnet_energy_force(
    model_obj, n_train, n_test, lr, epochs, dataset, batch_size
):
    pass


def train_schnet_force(model_obj, n_train, n_test, lr, epochs, dataset, batch_size):
    pass


# 'ISO17_energy_and_force': {'success': False},
# 'MD17_energy_and_force_aspirin': {'success': False},
# 'MD17_energy_and_force_azobenzene': {'success': False},
# 'MD17_energy_and_force_benzene': {'success': False},
# 'MD17_energy_and_force_ethanol': {'success': False},
# 'MD17_energy_and_force_malonaldehyde': {'success': False},
# 'MD17_energy_and_force_naphthalene': {'success': False},
# 'MD17_energy_and_force_paracetamol': {'success': False},
# 'MD17_energy_and_force_salicylic_acid': {'success': False},
# 'MD17_energy_and_force_toluene': {'success': False},
# 'MD17_energy_and_force_uracil': {'success': False},
