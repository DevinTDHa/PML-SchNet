import numpy as np
import pandas as pd
import plotly.express as px
import torch
from schnetpack import properties
from torch import nn
from tqdm import tqdm

from src.data_loader import load_data
from src.loss import energy_force_loss

label = "energy_U0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BaselineModel(nn.Module):
    def __init__(self, max_embeddings=100, embedding_dim=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(max_embeddings, embedding_dim),
            nn.Linear(embedding_dim, 1, bias=False),
        )

    def forward(self, input):
        n_atoms = input["N"]
        Z = input["Z"]

        y = self.model(Z)
        Y_batch = torch.split(
            y,
            n_atoms.view(
                len(n_atoms),
            ).tolist(),
        )
        batch_means = torch.tensor([pred.sum() for pred in Y_batch])
        batch_means.requires_grad_()
        return batch_means


def train(model, dataset, epochs=50, lr=0.01):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    losses = []
    model.train()
    for epoch in tqdm(range(epochs)):
        train_gen, test_gen = load_data(dataset, 100, 100)
        loss = None
        for X_batch, y_batch in train_gen:
            # Forward pass
            loss = criterion(model(X_batch), y_batch)
            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights
        print(f"Epoch {epoch + 1}, Train Loss: {loss:.4f}")
        losses.append({"epoch": epoch, "loss": loss.item()})
    plot_loss(losses)
    return losses


def validate_qm9(model, dataset):
    # Validation step
    criterion = nn.MSELoss()

    model.eval()
    with torch.no_grad():
        val_loss = []
        for data in dataset.val_dataloader():
            X_batch = data_to_dic(data)
            y_batch = data[label].float()
            val_loss.append(criterion(model(X_batch), y_batch).item())
    mean_loss = torch.Tensor(val_loss).mean()
    return mean_loss


def plot_loss(losses):
    fig = px.line(pd.DataFrame(losses), x="epoch", y="loss", title="Loss over epoch")
    fig.show()


def data_to_dic(x):
    return {
        "Z": x[properties.Z],  # nuclear charge, `Z` is `_atomic_numbers`
        "R": x[properties.position],  # atomic positions `R` is `_positions`
        "N": x[properties.n_atoms],
    }


class BaselineModelMD17AspirinEnergy(nn.Module):
    def __init__(self, embedding_dim=63):
        super().__init__()
        self.model = nn.Sequential(nn.Linear(embedding_dim, 1, bias=False))

    def forward(self, R):
        y = self.model(R.float())
        return y


class BaselineModelMD17AspirinEnergyForce(nn.Module):
    @staticmethod
    def LinearBatchReLu(input_dim, output_dim):
        return [
            nn.Linear(input_dim, output_dim, bias=False),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
        ]

    def __init__(self, embedding_dim=63):
        super().__init__()
        self.model = nn.Sequential(
            *self.LinearBatchReLu(embedding_dim, 256),
            *self.LinearBatchReLu(256, 1024),
            *self.LinearBatchReLu(1024, 256),
            nn.Linear(256, 1),
        )

    def forward(self, R):
        # n_atoms = input["N"]
        # Z = input["Z"]
        y = self.model(R.float())
        return y


def train_md17(model, dataset_iterator, lr=0.01):
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    label = "energy"

    # Specific for Aspirin
    E_mu, E_std = (torch.tensor(-406737.276528638), torch.tensor(5.94684041516964))
    # Training loop
    losses = []
    # with tqdm(total=len(dataset_iterator), unit="batch") as pbar:
    with tqdm(unit="batch") as pbar:
        for data, y_batch in dataset_iterator:
            batch_size = len(data["N"])
            y_batch = (y_batch.view(-1, 1) - E_mu) / E_std
            X_batch = data["R"].view(batch_size, -1)
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            # Forward pass
            loss = criterion(
                model(X_batch), y_batch.float()
            )  # Ensure y_batch is the correct shape
            losses.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            pbar.set_postfix({"Loss": loss.item()})
            pbar.update()
    return losses


def validate_md17(model, dataset_iterator, criterion=nn.MSELoss()):
    # Validation step
    model.eval()
    # with torch.no_grad():
    E_mu, E_std = (torch.tensor(-406737.276528638), torch.tensor(5.94684041516964))

    val_loss = []
    for data, y_batch in tqdm(dataset_iterator, unit="batch"):
        batch_size = len(data["N"])
        y_batch = (y_batch.view(-1, 1) - E_mu) / E_std
        X_batch = data["R"].view(batch_size, -1)
        X_batch, y_batch = (
            X_batch.to(device),
            y_batch.to(device),
        )

        # Forward pass
        E_pred = model(X_batch)
        E = y_batch.float()
        loss = criterion(E, E_pred)
        val_loss.append(loss.item())

    return np.array(val_loss).mean()


def train_md17_energy_force(model, dataset_iterator):
    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=100)

    # Training loop
    losses = []
    with tqdm(unit="batch") as pbar:
        for data, E in dataset_iterator:
            batch_size = len(data["N"])
            E = E.view(-1, 1)
            X_batch = data["R"].view(batch_size, -1)
            F_batch = data["F"].view(batch_size, -1)
            X_batch, E, F_batch = (
                X_batch.to(device),
                E.to(device),
                F_batch.to(device),
            )
            X_batch.requires_grad = True

            # Forward pass
            E_pred = model(X_batch)
            loss = energy_force_loss(
                E_pred, X_batch, E, F_batch
            )  # Ensure y_batch is the correct shape
            losses.append(loss.item())

            # Backward pass and optimization
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Compute gradients
            optimizer.step()  # Update weights

            pbar.set_postfix({"Loss": loss.item()})
            pbar.update()


def validate_md17_energy_force(model, dataset_iterator):
    # Validation step
    model.eval()
    # with torch.no_grad():
    val_loss = []
    for data, y_batch in tqdm(dataset_iterator):
        batch_size = len(data["N"])
        y_batch = y_batch.view(-1, 1)
        X_batch = data["R"].view(batch_size, -1)
        X_batch.requires_grad_()
        F_batch = data["F"].view(batch_size, -1)
        X_batch, y_batch, F_batch = (
            X_batch.to(device),
            y_batch.to(device),
            F_batch.to(device),
        )

        # Forward pass
        E_pred = model(X_batch)
        E = y_batch.float()
        loss = energy_force_loss(
            E_pred, X_batch, E, F_batch
        )  # Ensure y_batch is the correct shape
        val_loss.append(loss.item())
    # print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
    np.array(val_loss).mean()
