import os

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ExponentialLR
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


if __name__ == "__main__":
    model = SchNet().to(device)

    total_data = 404000
    n_train = int(total_data * 0.9)
    n_test = int(total_data * 0.1)
    batch_size = 32

    epochs = 5
    lr = 0.001
    scheduler_step_size = 100000
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.96, verbose=True)
    steps = 0
    criterion = nn.L1Loss()
    losses = []
    with tqdm(total=epochs, ncols=80, miniters=1, mininterval=1) as progress_bar:
        progress_bar.set_description("Schnet ISO17 E")
        for epoch in range(epochs):
            train_gen, test_gen = load_data(
                "ISO17",
                n_train,
                n_test,
                batch_size=batch_size,
                split_file="test_ref_train_iso17_long.split",
            )
            for X_batch, y_batch in train_gen:
                # Forward pass
                pred = model(X_batch)
                loss = criterion(pred, y_batch)

                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                progress_bar.set_postfix(train_loss=f"{loss:.4E}", refresh=False)
                losses.append(loss.item())

                steps += 1
                if steps % 100 == 0:
                    progress_bar.refresh()
                if steps == scheduler_step_size:
                    scheduler.step()
                    steps = 0

            # Validation Losses
            # val_losses = validate_schnet_energy(model, test_gen)
            # print(f"Validation Loss Epoch {epoch}", np.mean(val_losses))
            print(f"End of Epoch {epoch}, last train loss:", losses[-1])
            progress_bar.update(1)
            scheduler.step()

    save_model(model, "schnet_iso17_e")
    np.save("iso17_long_train_losses_e.npy", arr=losses)
    # np.save("iso17_long_val_losses.npy", arr=val_losses)
