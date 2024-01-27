import datetime
import sys

from ray.tune.schedulers import ASHAScheduler

sys.path.append(".")
from pml_schnet.settings import device

from pml_schnet.data_loader import load_data
from pml_schnet.training import (
    train_schnet_energy_force_hyperparam,
)
from pml_schnet.model import SchNet
from pml_schnet.activation import ShiftedSoftPlus
import torch
from torch.nn import GELU, LeakyReLU
import os
import numpy as np
from ray import tune
from ray.tune import CLIReporter


# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTrainingReplay.html#ray.tune.schedulers.PopulationBasedTrainingReplay
# https://openreview.net/forum?id=S1Y7OOlRZ
# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
# https://docs.ray.io/en/latest/tune/tutorials/tune_get_data_in_and_out.html
def save_model(model, path):
    torch.save(model, path + ".pt")


if __name__ == "__main__":
    print("Running on device", device)
    n_train = 10_000
    n_test = 1_000

    batch_size = 32
    epochs = 100
    lr = 1e-3

    cwd = os.getcwd()
    print("Current working directory is", cwd)
    run_name = "hyperparam_gridsearch"

    # num_samples = 300
    # print(f"Number of samplings: {num_samples}")

    logs_dir = f"{cwd}/ms3_hyperparam/{run_name}"

    def objective(config, data):
        atom_embedding_dim = int(config["atom_embedding_dim"])
        n_interactions = int(config["n_interactions"])
        rbf_max = config["rbf_max"]
        n_rbf = int(config["n_rbf"])
        activation_function = config["activation"]
        identifier_string = (
            f"{atom_embedding_dim}_{n_interactions}_{rbf_max}_{n_rbf}_{lr}_{activation_function}".replace(
                "=", "_"
            )
            .replace(",", "_")
            .replace("'", "_")
            .replace(" ", "")
        )
        activation_functions = {
            "ShiftedSoftPlus": ShiftedSoftPlus,
            "GELU": GELU,
            "LeakyReLU": LeakyReLU,
        }

        model = SchNet(
            atom_embedding_dim=atom_embedding_dim,
            n_interactions=n_interactions,
            rbf_max=rbf_max,
            n_rbf=n_rbf,
            activation=activation_functions[activation_function],
        )

        train_set, test_set = data
        losses, val_losses = train_schnet_energy_force_hyperparam(
            train_set=train_set,
            test_set=test_set,
            model=model,
            lr=lr,
            epochs=epochs,
            save_checkpoint=False,
        )

        print("torch logs in, ", logs_dir)
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        save_model(model, f"{logs_dir}/{identifier_string}")
        np.savetxt(f"{logs_dir}/train_iso17_losses_{identifier_string}.txt", losses)
        np.savetxt(f"{logs_dir}/val_iso17_losses_{identifier_string}.txt", val_losses)

        return {"validation_loss": np.min(val_losses)}

    train_set, test_set = load_data(
        "ISO17",
        n_train,
        n_test,
        batch_size=batch_size,
        split_file=None,
        molecule="NA",
        keep_in_memory=True,
        cache_pickle=True,
    )

    ray_logs_dir = f"{cwd}/ray_out/{run_name}"
    print("ray logs in", ray_logs_dir)
    if not os.path.exists(ray_logs_dir):
        os.makedirs(ray_logs_dir)

    config = {
        "atom_embedding_dim": tune.grid_search([2**i for i in range(5, 9)]),
        "n_interactions": tune.grid_search([2**i for i in range(1, 4)]),
        "rbf_max": tune.grid_search([10 * i for i in range(2, 5)]),
        "n_rbf": tune.grid_search([100 * 2**i for i in range(0, 3)]),
        "activation": tune.grid_search(["ShiftedSoftPlus", "LeakyReLU", "GELU"]),
    }

    objective_with_resources = tune.with_resources(objective, {"cpu": 1, "gpu": 1})
    analysis = tune.run(
        tune.with_parameters(objective_with_resources, data=(train_set, test_set)),
        config=config,
        metric="validation_loss",
        mode="min",
        scheduler=ASHAScheduler(),
        progress_reporter=CLIReporter(metric_columns=["validation_loss"]),
        local_dir=ray_logs_dir,
        log_to_file=True,
        time_budget_s=datetime.timedelta(days=1.8),
    )

    analysis.dataframe().to_csv(f"{logs_dir}/{run_name}_analysis.csv")
    best_config = analysis.get_best_config(metric="validation_loss", mode="min")
    print("Best parameters:", best_config)
