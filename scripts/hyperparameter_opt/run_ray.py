import os
import sys


sys.path.append(".")
from pml_schnet.settings import device

from pml_schnet.data_loader import load_data
from pml_schnet.training import (
    train_schnet_energy_force_data_loaded,
)
from pml_schnet.model import SchNet
from pml_schnet.activation import ShiftedSoftPlus
import torch
from torch.nn import GELU, LeakyReLU
import os
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTrainingReplay.html#ray.tune.schedulers.PopulationBasedTrainingReplay
# https://openreview.net/forum?id=S1Y7OOlRZ
# https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html
def save_model(model, path):
    torch.save(model, path + ".pt")


if __name__ == "__main__":
    print("Running on device", device)
    n_train = 50_000
    n_test = 1_000

    batch_size = 32
    epochs = 100

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

    def objective(config):
        atom_embedding_dim = int(config["atom_embedding_dim"])
        n_interactions = int(config["n_interactions"])
        rbf_max = config["rbf_max"]
        n_rbf = int(config["n_rbf"])
        lr = config["lr"]
        run_name = config["run_name"]
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

        losses, val_losses = train_schnet_energy_force_data_loaded(
            train_set=train_set,
            test_set=test_set,
            model=model,
            lr=lr,
            epochs=epochs,
            save_checkpoint=False,
        )

        logs_dir = f"/media/ckl/dump1/Documents/uni/WS_23_24/pml/pml_code/ms3_hyperparam/{run_name}"
        print("torch logs in, ", logs_dir)
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
        save_model(model, f"{logs_dir}/{identifier_string}")
        np.savetxt(f"{logs_dir}/train_iso17_losses_{identifier_string}.txt", losses)
        np.savetxt(f"{logs_dir}/val_iso17_losses_{identifier_string}.txt", val_losses)
        # import ray

        # ray.train.report({'LOSS': np.min(val_losses)})
        return {"LOSS": np.min(val_losses)}

    run_name = "THE_BIG_RUN"
    config = {
        # 'atom_embedding_dim': tune.quniform(32, 128, 1),
        # 'n_interactions': tune.quniform(1, 5, 1),
        # 'rbf_max': tune.uniform(20.0, 40.0),
        # 'n_rbf': tune.quniform(200, 400, 1),
        # 'lr': tune.loguniform(1e-4, 1e-2),
        "atom_embedding_dim": tune.choice([2**i for i in range(5, 9)]),
        "n_interactions": tune.choice([i**2 for i in range(1, 4)]),
        "rbf_max": tune.choice([10 * i for i in range(3, 7)]),
        "n_rbf": tune.choice([100 * i for i in range(1, 6)]),
        "lr": tune.choice([1e-3]),
        "activation": tune.choice(["ShiftedSoftPlus", "LeakyReLU", "GELU"]),
        "run_name": tune.choice([run_name]),
    }
    ray_logs_dir = (
        f"/media/ckl/dump1/Documents/uni/WS_23_24/pml/pml_code/ray_out/{run_name}"
    )
    print("ray logs in", ray_logs_dir)
    if not os.path.exists(ray_logs_dir):
        os.makedirs(ray_logs_dir)
    analysis = tune.run(
        objective,
        config=config,
        num_samples=4,
        scheduler=ASHAScheduler(metric="LOSS", mode="min"),
        progress_reporter=CLIReporter(
            metric_columns=["LOSS", "losses", "val_loss", "epoch"]
        ),
        local_dir=ray_logs_dir,
        log_to_file=True,
    )

    best_config = analysis.get_best_config(metric="min-LOSS", mode="min")
    print("Best parameters:", best_config)
