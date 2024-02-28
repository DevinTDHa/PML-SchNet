from dataclasses import dataclass

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Task:
    energy = "energy"
    force = "force"
    energy_and_force = "energy_and_force"


class Dataset:
    qm9 = "QM9"
    md17 = "MD17"
    iso17 = "ISO17"


class Model:
    baseline = "baseline"
    schnet = "schnet"


@dataclass
class Trainable:
    dataset: str
    task: str
    molecule: str = None

    def __str__(self):
        if self.molecule:
            return f"{self.dataset}_{self.task}_{self.molecule}"
        else:
            return f"{self.dataset}_{self.task}"


valid_molecules = {
    "aspirin": 21,
    "azobenzene": 24,
    "benzene": 12,
    "ethanol": 9,
    "malonaldehyde": 9,
    "naphthalene": 18,
    "paracetamol": 20,
    "salicylic_acid": 16,
    "toluene": 15,
    "uracil": 12,
}



md17_trainable_all = (
    [Trainable(Dataset.md17, Task.energy, m) for m in valid_molecules]
    + [Trainable(Dataset.md17, Task.force, m) for m in valid_molecules]
    + [Trainable(Dataset.md17, Task.energy_and_force, m) for m in valid_molecules]
)

md17_trainable_one_molecule = [
    Trainable(Dataset.md17, Task.energy, "aspirin"),
    Trainable(Dataset.md17, Task.force, "aspirin"),
    Trainable(Dataset.md17, Task.energy_and_force, "aspirin"),
]

iso17_trainable = [
    Trainable(Dataset.iso17, Task.energy),
    Trainable(Dataset.iso17, Task.force),
    Trainable(Dataset.iso17, Task.energy_and_force),
]
qm9_trainable = [Trainable(Dataset.qm9, "energy")]

all_trainable = md17_trainable_all + iso17_trainable + qm9_trainable
all_trainable_one_molecule = (
    md17_trainable_one_molecule + iso17_trainable + qm9_trainable
)

train_modes = {
    "md17_all_molecules": md17_trainable_all,
    "md17_one_molecule": md17_trainable_one_molecule,
    "iso17": iso17_trainable,
    "qm9": qm9_trainable,
    "all": all_trainable,
    "all_one_molecule": all_trainable_one_molecule,
}

# cache_dir = "./" if on_dev_machine() else : "/home/space/datasets/schnet"

cache_dir = "./"
