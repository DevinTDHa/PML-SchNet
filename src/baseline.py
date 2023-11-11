import pandas as pd
import plotly.express as px
import torch
from schnetpack import properties
from torch import nn

from src.data_loader import load_data


class BaselineModel(nn.Module):
    def __init__(self, max_embeddings=100, embedding_dim=8):
        super().__init__()
        self.model = nn.Sequential(
            nn.Embedding(max_embeddings, embedding_dim),
            nn.Linear(embedding_dim, 1, bias=False)
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
    label = 'energy_U0'
    losses = []
    model.train()
    for epoch in range(epochs):
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
        losses.append({'epoch': epoch, 'loss': loss.item()})
    plot_loss(losses)


def validate():
    raise NotImplemented("todo")
    # # Validation step
    # model.eval()
    # with torch.no_grad():
    #     val_loss = 0
    #     for data in dataset.val_dataloader():
    #         X_batch = data_to_dic(data)
    #         y_batch = data[label].float()
    #         val_loss += criterion(model(X_batch), y_batch).item()


def plot_loss(losses):
    fig = px.line(pd.DataFrame(losses), x="epoch", y="loss", title='Loss over epoch')
    fig.show()


def data_to_dic(x):
    return {
        'Z': x[properties.Z],  # nuclear charge, `Z` is `_atomic_numbers`
        'R': x[properties.position],  # atomic positions `R` is `_positions`
        'N': x[properties.n_atoms]
    }
