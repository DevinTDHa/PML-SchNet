import os

import numpy as np
from schnetpack import properties
from schnetpack.datasets import ISO17, QM9, MD17
from schnetpack.transform import ASENeighborList

from pml_schnet import settings
from pml_schnet.data_utils import fix_iso_17_db
from pml_schnet.settings import valid_molecules, device

energy_label = {"QM9": "energy_U0", "ISO17": "total_energy", "MD17": "energy"}
force_label = {"QM9": None, "MD17": "forces", "ISO17": "atomic_forces"}


def data_to_dic(x, dataset):
    inputs = {
        "Z": x[properties.Z].to(device),  # nuclear charge, `Z` is `_atomic_numbers`
        "R": x[properties.position]
        .to(device)
        .float(),  # atomic positions `R` is `_positions`
        "N": x[properties.n_atoms].tolist(),  # Number of atoms
        "idx_i": x[properties.idx_i].to(device),  # Index of first atom for distance
        "idx_j": x[properties.idx_j].to(device),  # Index of second atom for distance
    }
    if dataset != "QM9":
        inputs["F"] = x[force_label[dataset]].to(device)

    return inputs


def get_generator(base_gen, dataset):
    for x in base_gen:
        yield data_to_dic(x, dataset), x[energy_label[dataset]].float().to(device)


def load_data(
    dataset="QM9",
    n_train=1000,
    n_test=100,
    batch_size=32,
    molecule="aspirin",
    log=False,
    iso17_fold="reference",
    cache_dir=settings.cache_dir,
    split_file=None,
):
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if dataset == "QM9":
        data = QM9(
            os.path.join(cache_dir, "qm9.db"),
            batch_size=batch_size,
            num_train=n_train,
            num_test=0,
            num_val=n_test,
            transforms=[ASENeighborList(np.inf)],
            split_file=split_file,
        )
    elif dataset == "MD17":
        if molecule is None:
            raise Exception("Please specify a molecule")
        elif molecule not in valid_molecules:
            raise Exception(
                f"Please choose one of the following molecules : {valid_molecules}"
            )
        data = MD17(
            os.path.join(cache_dir, f"md17_{molecule}.db"),
            # fold = 'reference', # !! new param
            molecule=molecule,
            batch_size=batch_size,
            num_train=n_train,
            num_test=0,
            num_val=n_test,
            transforms=[ASENeighborList(np.inf)],
            split_file=split_file,
        )
    elif dataset == "ISO17":
        db_path = os.path.join(cache_dir, "iso17.db")
        fix_iso_17_db(db_path)
        data = ISO17(
            db_path,
            fold=iso17_fold,
            batch_size=batch_size,
            num_train=n_train,
            num_test=0,
            num_val=n_test,
            transforms=[ASENeighborList(np.inf)],
            split_file=split_file,
        )
    else:
        raise ValueError("Only QM9, MD17 and ISO17 are supported but used ", dataset)

    data.prepare_data()
    data.setup()

    test = data.val_dataloader()
    train = data.train_dataloader()

    if log:
        print("Number of reference calculations:", len(data.dataset))
        print("Number of train data:", len(data.train_dataset))
        print("Number of validation data:", len(data.val_dataset))
        print("Number of test data:", len(data.test_dataset))
        print("Available properties:")

        for p in data.dataset.available_properties:
            print("-", p)
    return get_generator(train, dataset), get_generator(test, dataset)
