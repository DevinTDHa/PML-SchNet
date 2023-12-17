import torch
from torch import nn

from pml_schnet.data_loader import load_data
from pml_schnet.loss import derive_force, energy_force_loss
from pml_schnet.settings import device


def validate_baseline_energy(model, dataset, n_train, n_test, molecule):
    # Validation step
    criterion = nn.L1Loss()
    train_gen, test_gen = load_data(
        dataset, n_train=n_train, n_test=n_test, molecule=molecule
    )
    model.eval()
    with torch.no_grad():
        val_loss = []
        for X_batch, y_batch in train_gen:
            X_batch["N"] = X_batch["N"].to(device)
            X_batch["Z"] = X_batch["Z"].to(device)
            X_batch["R"] = X_batch["R"].to(device)
            y_batch = y_batch.to(device)
            val_loss.append(criterion(model(X_batch), y_batch).item())
    mean_loss = torch.Tensor(val_loss).mean()
    return float(mean_loss.numpy())


def validate_baseline_force(model, dataset, n_train, n_test, molecule):
    # Validation step
    criterion = nn.L1Loss()
    train_gen, test_gen = load_data(
        dataset, n_train=n_train, n_test=n_test, molecule=molecule
    )
    model.eval()

    val_loss = []
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
        val_loss.append(loss.item())
    mean_loss = torch.Tensor(val_loss).mean()
    return float(mean_loss.numpy())


def validate_baseline_energy_force(model, dataset, n_train, n_test, molecule):
    # Validation step
    train_gen, test_gen = load_data(
        dataset, n_train=n_train, n_test=n_test, molecule=molecule
    )
    model.eval()
    val_loss = []
    for X_batch, y_batch in train_gen:
        X_batch["N"] = X_batch["N"].to(device)
        X_batch["Z"] = X_batch["Z"].to(device)
        X_batch["R"] = X_batch["R"].to(device)

        X_batch["R"].requires_grad_()
        F = X_batch["F"].to(device)
        y_batch = y_batch.to(device)
        E_pred = model(X_batch)
        loss = energy_force_loss(E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F)

        val_loss.append(loss.item())
    mean_loss = torch.Tensor(val_loss).mean().item()
    return mean_loss


def validate_schnet_energy(model, dataset, n_train, n_test, molecule):
    pass


def validate_schnet_force(model, dataset, n_train, n_test, molecule):
    pass


def validate_schnet_energy_force(model, dataset, n_train, n_test, molecule):
    pass
