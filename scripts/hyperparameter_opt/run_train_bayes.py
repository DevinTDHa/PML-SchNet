import os

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

from activation import ShiftedSoftPlus, ReLU
from model import SchNet
from test.iso17_train.iso17_train_ef import (
    save_model,
)  # Replace with your actual module path
from training import train_schnet_energy_force


# Define the objective function for Hyperopt
def objective(params):
    atom_embedding_dim = int(params["atom_embedding_dim"])
    n_interactions = int(params["n_interactions"])
    # max_z = int(params["max_z"])
    # rbf_min = params["rbf_min"]
    rbf_max = params["rbf_max"]
    n_rbf = int(params["n_rbf"])
    lr = params["lr"]
    activation_function = params["activation"]
    # Map string to actual activation function
    activation_functions = {
        "ShiftedSoftPlus": ShiftedSoftPlus,
        "ReLU": ReLU,  # Replace with actual ReLU function if different
    }

    model = SchNet(
        atom_embedding_dim=atom_embedding_dim,
        n_interactions=n_interactions,
        # max_z=max_z,
        # rbf_min=rbf_min,
        rbf_max=rbf_max,
        n_rbf=n_rbf,
        activation=activation_functions[activation_function],
    )  # .to("cuda")

    total_data = 10
    n_train = int(total_data * 0.9)
    n_test = int(total_data * 0.1)

    batch_size = 32
    epochs = 30  # Adjust as needed

    if os.path.exists("iso17_confidence_split.npz"):
        os.remove("iso17_confidence_split.npz")

    losses, val_losses = train_schnet_energy_force(
        model_obj=model,
        n_train=n_train,
        n_test=n_test,
        lr=lr,
        epochs=epochs,
        dataset="ISO17",
        batch_size=batch_size,
        molecule="NA",
        split_file="iso17_confidence_split",
    )

    save_model(model, "iso17_confidence_model")
    np.savetxt("iso17_confidence_train_losses.txt", losses)
    np.savetxt("iso17_confidence_val_losses.txt", val_losses)

    # Hyperopt minimizes the objective; return the validation loss as a metric
    return {"loss": np.min(val_losses), "status": STATUS_OK}


# Define the hyperparameter space
space = {
    "atom_embedding_dim": hp.quniform("atom_embedding_dim", 32, 128, 1),
    "n_interactions": hp.quniform("n_interactions", 1, 5, 1),
    # 'max_z': hp.quniform('max_z', 50, 150, 1),
    # 'rbf_min': hp.uniform('rbf_min', 0.0, 1.0),
    "rbf_max": hp.uniform("rbf_max", 20.0, 40.0),
    "n_rbf": hp.quniform("n_rbf", 200, 400, 1),
    "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-2)),
    "activation": hp.choice("activation", ["ShiftedSoftPlus", "ReLU"]),
}

if __name__ == "__main__":
    # Run the optimization
    trials = Trials()
    best = fmin(
        fn=objective, space=space, algo=tpe.suggest, max_evals=20, trials=trials
    )

    print("Best parameters:", best)
