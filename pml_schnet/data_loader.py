import os
import pickle

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
        .float()
        .to(device),  # atomic positions `R` is `_positions`
        "N": x[properties.n_atoms].tolist(),  # Number of atoms
        "idx_i": x[properties.idx_i].to(device),  # Index of first atom for distance
        "idx_j": x[properties.idx_j].to(device),  # Index of second atom for distance
    }
    if dataset != "QM9":
        inputs["F"] = x[force_label[dataset]].float().to(device)

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
    keep_in_memory=False,
    cache_pickle=False,
):
    """
    Load data from the specified dataset with the given parameters and settings.

    Args:
        dataset (str): The name of the dataset to load.
        n_train (int): The number of training samples to load.
        n_test (int): The number of test samples to load.
        batch_size (int): The batch size for loading the data.
        molecule (str): The name of the molecule to load (default: "aspirin").
        log (bool): Whether to log information during data loading (default: False).
        iso17_fold (str): The fold for ISO17 dataset (default: "reference").
        cache_dir (str): The directory to cache the data (default: settings.cache_dir).
        split_file (str): The file to use for data splitting (default: None).
        keep_in_memory (bool): Whether to keep the loaded data in memory (default: False).
        cache_pickle (bool): Whether to cache the data using pickle (default: False).

    Returns:
        tuple or generator: Depending on the settings, returns a tuple or generator
        containing the loaded training and test data.
    """
    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if dataset == "QM9":
        data = QM9(
            os.path.join(cache_dir, "qm9.db"),
            batch_size=batch_size,
            num_train=n_train,
            num_test=0,
            num_val=n_test,
            transforms=[ASENeighborList(10e6)],
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
            transforms=[ASENeighborList(1e6)],
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
            transforms=[ASENeighborList(10e6)],
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

    if keep_in_memory:
        print("Loading data into memory...")
        train_pkl = f"{dataset}_{n_train}_{n_test}_{batch_size}_train.pkl"
        train_path = os.path.join(cache_dir, train_pkl)
        test_pkl = f"{dataset}_{n_train}_{n_test}_{batch_size}_test.pkl"
        test_path = os.path.join(cache_dir, test_pkl)
        if cache_pickle:
            if os.path.exists(train_path) and os.path.exists(test_path):
                print("Loading data from cache...")
                with open(train_path, "rb") as f:
                    train_set = pickle.load(f)
                with open(test_path, "rb") as f:
                    test_set = pickle.load(f)
            else:
                train_set, test_set = list(get_generator(train, dataset)), list(
                    get_generator(test, dataset)
                )
                print("Caching data as pickle.")
                with open(train_path, "wb") as f:
                    pickle.dump(train_set, f)
                with open(test_path, "wb") as f:
                    pickle.dump(test_set, f)
        else:
            train_set, test_set = list(get_generator(train, dataset)), list(
                get_generator(test, dataset)
            )
        return train_set, test_set
    else:
        return get_generator(train, dataset), get_generator(test, dataset)
