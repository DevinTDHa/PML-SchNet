import os

import numpy as np
import ray
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from pml_schnet.data_loader import load_data
from pml_schnet.loss import derive_force, energy_force_loss, energy_force_loss_mae
from pml_schnet.model import SchNet
from pml_schnet.settings import device, get_device
from pml_schnet.visualization.plotting import plot_loss


def write_grads(model, writer, epoch):
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            # print(f'name={name}, shape={param.grad.shape}')
            writer.add_histogram(f"Gradients/{name}", param.grad, epoch)
            writer.add_scalar(f"Gradient_norm/{name}", param.grad.norm(), epoch)


def validate_schnet_energy(model, test_gen, criterion):
    val_loss = []
    labels = []
    for X_batch, y_batch in test_gen:
        # Forward pass
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        # labels.append(pred.item())
        labels.append(pred.detach().to("cpu"))
        val_loss.append(loss.item())

    return np.mean(val_loss), labels


def validate_schnet_force(model, test_gen, criterion):
    val_loss = []
    labels = []
    for X_batch, y_batch in test_gen:
        # Forward pass
        X_batch["R"].requires_grad_()
        target_F = X_batch["F"]
        target_F.requires_grad_()

        E_pred = model(X_batch)
        F_pred = derive_force(E_pred, X_batch["R"])

        loss = criterion(F_pred, target_F)
        # labels.append(pred.item())
        labels.append(F_pred.detach().to("cpu"))
        val_loss.append(loss.item())

    return np.mean(val_loss), labels


# def validate_schnet_force_energy(model, test_gen):
#     val_loss = []
#     labels = []
#     for X_batch, y_batch in test_gen:
#         # Forward pass
#         X_batch["R"].requires_grad_()
#         F = X_batch["F"].to(device)
#
#         # Forward pass
#         E_pred = model(X_batch)
#         loss = energy_force_loss(E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F)
#         E_pred.detach()
#         loss.detach()
#         labels.append(E_pred.detach().to("cpu"))
#
#         val_loss.append(loss.item())
#
#     return np.mean(val_loss), labels


def validate_schnet_force_energy(model, test_gen, output_labels=False):
    model.eval()
    val_loss = []
    labels = []
    for X_batch, y_batch in test_gen:
        # Forward pass
        X_batch["Z"] = X_batch["Z"].to(device)
        X_batch["idx_i"] = X_batch["idx_i"].to(device)
        X_batch["idx_j"] = X_batch["idx_j"].to(device)
        X_batch["R"] = X_batch["R"].to(device)
        X_batch["R"].to(device).requires_grad_()
        F = X_batch["F"].to(device)
        y_batch = y_batch.to(device)

        # Forward pass
        E_pred = model(X_batch)
        loss = energy_force_loss_mae(E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F)
        E_pred.detach()
        loss.detach()
        if output_labels:
            labels.append(E_pred.detach().to("cpu"))

        val_loss.append(loss.item())

    model.train()
    return np.mean(val_loss), labels


def train_baseline_energy(model, n_train, n_test, lr, epochs, dataset):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    losses = []
    for epoch in tqdm(range(epochs)):
        train_gen, test_gen = load_data(dataset, n_train, n_test)
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
    model_obj,
    n_train,
    n_test,
    lr,
    epochs,
    dataset,
    batch_size,
    molecule,
    split_file=None,
    scheduler_step_size=100_000,
):
    # writer = model_obj.writer
    optimizer = torch.optim.Adam(model_obj.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.96, verbose=True)
    steps = 0
    criterion = nn.L1Loss()
    losses = []
    val_losses = []
    with tqdm(total=epochs, ncols=80) as progress_bar:
        progress_bar.set_description("Schnet E")
        for epoch in range(epochs):
            train_gen, test_gen = load_data(
                dataset,
                n_train,
                n_test,
                batch_size=batch_size,
                molecule=molecule,
                split_file=split_file,
            )
            for X_batch, y_batch in train_gen:
                # Forward pass
                pred = model_obj(X_batch)
                loss = criterion(pred, y_batch)
                losses.append(loss.item())
                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                update_pbar_desc(loss, progress_bar, steps)
                steps += 1
                if steps == scheduler_step_size:
                    scheduler.step()
                    steps = 0

            # End of Epoch
            val_loss, _ = validate_schnet_energy(model_obj, test_gen, criterion)
            val_losses.append(val_loss)
            progress_bar.update(1)
    return np.array(losses), np.array(val_losses)


def train_schnet_energy_force(
    model_obj: SchNet,
    n_train,
    n_test,
    lr,
    epochs,
    dataset,
    batch_size,
    molecule,
    split_file=None,
    scheduler_step_size=100_000,
):
    # writer = model_obj.writer
    optimizer = torch.optim.Adam(model_obj.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.96, verbose=True)
    steps = 0
    criterion = energy_force_loss
    losses = []
    val_losses = []
    with tqdm(total=epochs, ncols=80) as progress_bar:
        progress_bar.set_description("Schnet E+F")
        for epoch in range(epochs):
            train_gen, test_gen = load_data(
                dataset,
                n_train,
                n_test,
                batch_size=batch_size,
                split_file=split_file,
                molecule=molecule,
            )
            for X_batch, y_batch in train_gen:
                # Forward pass
                X_batch["R"].requires_grad_()
                F = X_batch["F"].to(device)

                # Only for the first epoch
                if epoch == 0 and model_obj.running_mean_var:
                    model_obj.update_mean_var(y_batch, X_batch["F"])

                # Forward pass
                E_pred = model_obj(X_batch)
                loss = criterion(E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F)
                losses.append(loss.item())

                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                update_pbar_desc(loss, progress_bar, steps)
                steps += 1
                if steps == scheduler_step_size:
                    scheduler.step()
                    steps = 0

            # End of Epoch
            val_loss, _ = validate_schnet_force_energy(model_obj, test_gen)
            val_losses.append(val_loss)
            progress_bar.update(1)
    return np.array(losses), np.array(val_losses)


def train_schnet_energy_force_mem(
    model: SchNet,
    n_train,
    n_test,
    lr,
    epochs,
    dataset,
    batch_size,
    molecule,
    split_file=None,
    scheduler_step_size=100_000,
    save_checkpoint=True,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.96, verbose=True)
    steps = 0
    criterion = energy_force_loss
    losses = []
    val_losses = []

    if save_checkpoint:
        os.makedirs("checkpoints", exist_ok=True)

    train_set, test_set = load_data(
        dataset,
        n_train,
        n_test,
        batch_size=batch_size,
        split_file=split_file,
        molecule=molecule,
        keep_in_memory=True,
        cache_pickle=True,
    )

    lowest_loss = np.inf
    with tqdm(total=epochs, ncols=80) as progress_bar:
        progress_bar.set_description("Schnet E+F")
        for epoch in range(epochs):
            for X_batch, y_batch in train_set:
                # Forward pass
                X_batch["R"].requires_grad_()
                F = X_batch["F"].to(device)

                # Only for the first epoch
                if epoch == 0 and model.running_mean_var:
                    model.update_mean_var(y_batch, X_batch["F"])

                # Forward pass
                E_pred = model(X_batch)
                loss = criterion(E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F)
                losses.append(loss.item())

                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                update_pbar_desc(loss, progress_bar, steps)
                steps += 1
                if steps == scheduler_step_size:
                    scheduler.step()
                    steps = 0

            # End of Epoch
            val_loss, _ = validate_schnet_force_energy(model, test_set)
            val_losses.append(val_loss)

            if val_loss < lowest_loss:
                lowest_loss = val_loss
                if save_checkpoint:
                    print("Saving checkpoint for loss: ", lowest_loss, epoch)
                    torch.save(
                        model,
                        f"checkpoints/schnet_ef_chkp_l{val_loss:.4f}_e{epoch}.pt",
                    )

            progress_bar.update(1)
    return np.array(losses), np.array(val_losses)


def train_schnet_energy_force_hyperparam(
    train_set,
    test_set,
    model: SchNet,
    lr,
    epochs,
    scheduler_step_size=100_000,
    save_checkpoint=False,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.96)
    steps = 0
    criterion = energy_force_loss
    losses = []
    val_losses = []

    device = get_device()
    model.to(device)

    if save_checkpoint:
        os.makedirs("checkpoints", exist_ok=True)

    lowest_loss = np.inf
    for epoch in range(epochs):
        for X_batch, y_batch in train_set:
            # Forward pass
            X_batch["Z"] = X_batch["Z"].to(device)
            X_batch["idx_i"] = X_batch["idx_i"].to(device)
            X_batch["idx_j"] = X_batch["idx_j"].to(device)
            X_batch["R"] = X_batch["R"].to(device)
            X_batch["R"].to(device).requires_grad_()
            F = X_batch["F"].to(device)
            y_batch = y_batch.to(device)

            # Forward pass
            E_pred = model(X_batch)
            loss = criterion(E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F)
            loss_item = loss.item()
            losses.append(loss_item)

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            steps += 1
            if steps == scheduler_step_size:
                scheduler.step()
                steps = 0

            # End of Epoch
            val_loss, _ = validate_schnet_force_energy(model, test_set)
            val_losses.append(val_loss)
            # Log all loss values for the current epoch
            ray.train.report(metrics={"validation_loss": val_loss})

            if val_loss < lowest_loss:
                lowest_loss = val_loss
                if save_checkpoint:
                    print("Saving checkpoint for loss: ", lowest_loss, epoch)
                    torch.save(
                        model,
                        f"checkpoints/schnet_ef_chkp_l{val_loss:.4f}_e{epoch}.pt",
                    )

    return np.array(losses), np.array(val_losses)


def train_schnet_force(
    model_obj,
    n_train,
    n_test,
    lr,
    epochs,
    dataset,
    batch_size,
    molecule,
    split_file=None,
    scheduler_step_size=100_000,
):
    # writer = model_obj.writer
    optimizer = torch.optim.Adam(model_obj.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.96, verbose=True)
    steps = 0
    criterion = nn.L1Loss()
    losses = []
    val_losses = []
    with tqdm(total=epochs, ncols=80) as progress_bar:
        progress_bar.set_description("Schnet F")
        for epoch in range(epochs):
            train_gen, test_gen = load_data(
                dataset,
                n_train,
                n_test,
                batch_size=batch_size,
                split_file=split_file,
                molecule=molecule,
            )
            for X_batch, y_batch in train_gen:
                # Forward pass
                X_batch["R"].requires_grad_()
                target_F = X_batch["F"]
                target_F.requires_grad_()

                E_pred = model_obj(X_batch)

                F_pred = derive_force(E_pred, X_batch["R"])

                loss = criterion(F_pred, target_F)
                losses.append(loss.item())
                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                update_pbar_desc(loss, progress_bar, steps)

                steps += 1
                if steps == scheduler_step_size:
                    scheduler.step()
                    steps = 0

            # End of Epoch
            val_loss, _ = validate_schnet_force(model_obj, test_gen, criterion)
            val_losses.append(val_loss)
            progress_bar.update(1)

    return np.array(losses), np.array(val_losses)


def update_pbar_desc(loss, progress_bar, steps):
    progress_bar.set_postfix(train_loss=f"{loss:.4E}", steps=str(steps), refresh=False)
    if steps % 100 == 0:
        progress_bar.refresh()
