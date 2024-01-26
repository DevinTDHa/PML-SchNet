import os
import sys
sys.path.append('.')
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from pml_schnet.training import train_schnet_energy_force
from pml_schnet.model import SchNet
from pml_schnet.activation import ShiftedSoftPlus, ReLU  # Replace with actual activation functions
import torch


# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTrainingReplay.html#ray.tune.schedulers.PopulationBasedTrainingReplay
# https://openreview.net/forum?id=S1Y7OOlRZ
# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
def save_model(model, path):

    torch.save(model, path + ".pt")

import os
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

def objective(config):
    atom_embedding_dim = int(config['atom_embedding_dim'])
    n_interactions = int(config['n_interactions'])
    rbf_max = config['rbf_max']
    n_rbf = int(config['n_rbf'])
    lr = config['lr']
    activation_function = config['activation']

    activation_functions = {
        'ShiftedSoftPlus': ShiftedSoftPlus,
        'ReLU': ReLU
    }

    model = SchNet(
        atom_embedding_dim=atom_embedding_dim,
        n_interactions=n_interactions,
        rbf_max=rbf_max,
        n_rbf=n_rbf,
        activation=activation_functions[activation_function],
    )

    total_data = 10
    n_train = int(total_data * 0.9)
    n_test = int(total_data * 0.1)

    batch_size = 32
    epochs = 30

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

    return np.min(val_losses)

if __name__ == "__main__":
    config = {
        'atom_embedding_dim': tune.quniform(32, 128, 1),
        'n_interactions': tune.quniform(1, 5, 1),
        'rbf_max': tune.uniform(20.0, 40.0),
        'n_rbf': tune.quniform(200, 400, 1),
        'lr': tune.loguniform(1e-4, 1e-2),
        'activation': tune.choice(['ShiftedSoftPlus', 'ReLU'])
    }

    analysis = tune.run(
        objective,
        config=config,
        num_samples=20,
        scheduler=ASHAScheduler(metric="min-LOSS", mode="min"),
        progress_reporter=CLIReporter(metric_columns=["LOSS"]),
    )

    best_config = analysis.get_best_config(metric="min-LOSS", mode="min")
    print("Best parameters:", best_config)
