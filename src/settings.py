from dataclasses import dataclass


class Task:
    energy = 'energy'
    force = 'force'


class Dataset:
    qm9 = 'QM9'
    md17 = 'MD17'
    iso17 = 'ISO17'


class Model:
    baseline = 'baseline'
    schnet = 'schnet'


@dataclass
class Trainable:
    dataset: str
    task: str
    molecule: str = None

valid_molecules = {
    "aspirin": 21,
    "azobenzene": 12,
    "benzene": 6,
    "ethanol": 9,
    "malonaldehyde": 9,
    "naphthalene": 10,
    "paracetamol": 22,
    "salicylic_acid": 14,
    "toluene": 7,
    "uracil": 12,
}


md17_trainable_all = [Trainable(Dataset.md17, Task.energy, m) for m in valid_molecules] + \
                     [Trainable(Dataset.md17, Task.force, m) for m in valid_molecules]
md17_trainable_one_molecule = [Trainable(Dataset.md17, Task.energy, 'aspirin'),
                               Trainable(Dataset.md17, Task.force, 'aspirin')]

iso17_trainable = [Trainable(Dataset.iso17, Task.energy), Trainable(Dataset.iso17, Task.force)]
qm9_trainable = [Trainable(Dataset.qm9, "energy")]

all_trainable = md17_trainable_all + iso17_trainable + qm9_trainable
all_trainable_one_molecule = md17_trainable_one_molecule + iso17_trainable + qm9_trainable

train_modes = {
    'md17_all_molecules': md17_trainable_all,
    'md17_one_molecule': md17_trainable_one_molecule,
    'iso17': iso17_trainable,
    'qm9': qm9_trainable,
    'all': all_trainable,
    'all_one_molecule': all_trainable_one_molecule,
}

cache_dir = '/home/space/datasets/schnet/'
