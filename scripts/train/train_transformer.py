import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch

from pml_schnet.model import SchNetTransformer
from pml_schnet.settings import device
from pml_schnet.training import train_schnet_energy_force_mem


def save_model(model, path):
    torch.save(model, path + ".pt")


if __name__ == "__main__":
    print("Running on device", device)
    model = SchNetTransformer(running_mean_var=False).to(device)

    total_data = 404_000
    n_train = int(total_data * 0.9)
    n_test = int(total_data * 0.1)
    lr = 1e-3

    batch_size = 32
    epochs = 150

    losses, val_losses = train_schnet_energy_force_mem(
        model=model,
        n_train=n_train,
        n_test=n_test,
        lr=lr,
        epochs=epochs,
        dataset="ISO17",
        batch_size=batch_size,
        molecule="NA",
        split_file="iso17_split",
    )

    save_model(model.to("cpu"), "iso17_transformer_model")
    np.savetxt("iso17_transformer_train_losses.txt", losses)
    np.savetxt("iso17_transformer_val_losses.txt", val_losses)
