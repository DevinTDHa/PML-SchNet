import os
import sys

sys.path.append(os.getcwd())

import shutil

import numpy as np
import torch

from pml_schnet.model import SchNetDropout
from pml_schnet.settings import device
from pml_schnet.training import train_schnet_energy_force_mem


def save_model(model, path):
    torch.save(model, path + ".pt")


if __name__ == "__main__":
    print("Running on device", device)
    model = SchNetDropout(running_mean_var=True).to(device)

    total_data = 404  # _000
    n_train = int(total_data * 0.9)
    n_test = int(total_data * 0.1)
    lr = 1e-3

    batch_size = 32
    epochs = 1  # 50

    shutil.rmtree("iso17_dropout_split.npz", ignore_errors=True)
    losses, val_losses = train_schnet_energy_force_mem(
        model=model,
        n_train=n_train,
        n_test=n_test,
        lr=lr,
        epochs=epochs,
        dataset="ISO17",
        batch_size=batch_size,
        molecule="NA",
        split_file="iso17_dropout_split",
    )

    save_model(model.to("cpu"), "iso17_dropout_model")
    np.savetxt("iso17_dropout_train_losses.txt", losses)
    np.savetxt("iso17_dropout_val_losses.txt", val_losses)
