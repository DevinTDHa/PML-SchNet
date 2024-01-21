import os

import numpy as np

from iso17_train.iso17_train_ef import save_model
from model import SchNet
from training import train_schnet_energy_force

if __name__ == "__main__":
    model = SchNet(running_mean_var=False).to("cuda")

    total_data = 404_000
    n_train = int(total_data * 0.9)
    n_test = int(total_data * 0.1)
    lr = 1e-3

    batch_size = 32
    epochs = 30  # 200 epochs -> 19 hours
    os.remove("iso17_confidence_split.npz")
    losses, val_losses = train_schnet_energy_force(
        model_obj=model,
        n_train=n_train,
        n_test=n_test,
        lr=lr,
        epochs=epochs,
        dataset="ISO17",
        batch_size=32,
        molecule="NA",
        split_file="iso17_confidence_split",
    )

    save_model(model, "iso17_confidence_model")
    np.savetxt("iso17_confidence_train_losses.txt", losses)
    np.savetxt("iso17_confidence_val_losses.txt", val_losses)
