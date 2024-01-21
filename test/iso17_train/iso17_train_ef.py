import numpy as np
import torch
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from pml_schnet.data_loader import load_data
from pml_schnet.loss import energy_force_loss
from pml_schnet.model import SchNet
from pml_schnet.settings import (
    device,
)


def save_model(model, path):
    torch.save(model, path + ".pt")


if __name__ == "__main__":
    """10000 train data points, 1000 test data points, in mini-batches of size 32"""
    model = SchNet().to(device)

    total_data = 404000
    n_train = int(total_data * 0.9)
    n_test = int(total_data * 0.1)
    batch_size = 32

    epochs = 5
    lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ExponentialLR(optimizer=optimizer, gamma=0.96, verbose=True)
    scheduler_step_size = 100_000
    steps = 0
    losses = []
    with tqdm(total=epochs, ncols=80) as progress_bar:
        progress_bar.set_description("Schnet ISO17 E+F")
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
                X_batch["R"].requires_grad_()
                F = X_batch["F"].to(device)

                # Forward pass
                E_pred = model(X_batch)
                loss = energy_force_loss(E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F)

                # Backward pass and optimization
                optimizer.zero_grad()  # Clear gradients
                loss.backward()  # Compute gradients
                optimizer.step()  # Update weights

                progress_bar.set_postfix(
                    train_loss=f"{loss:.4E}", steps=str(steps), refresh=False
                )
                losses.append(loss.item())

                steps += 1
                if steps % 100 == 0:
                    progress_bar.refresh()
                if steps == scheduler_step_size:
                    scheduler.step()
                    steps = 0

            # # Validation Losses
            # val_losses = validate_schnet_force_energy(model, test_gen)
            # print(f"Validation Loss Epoch {epoch}", np.mean(val_losses))
            progress_bar.update(1)

    save_model(model, "schnet_iso17_ef")
    np.save("iso17_long_train_losses_ef.npy", arr=losses)
