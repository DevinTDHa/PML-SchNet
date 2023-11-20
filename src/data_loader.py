import os

from schnetpack import properties
from schnetpack.datasets import ISO17, QM9, MD17
from schnetpack.transform import ASENeighborList

from src.data_utils import fix_iso_17_db

energy_label = {"QM9": "energy_U0", "ISO17": "total_energy", "MD17": "energy"}
force_label = {"QM9": None, "MD17": "forces", "ISO17": "atomic_forces"}


def data_to_dic(x, dataset):
    return {
        "Z": x[properties.Z],  # nuclear charge, `Z` is `_atomic_numbers`
        "R": x[properties.position],  # atomic positions `R` is `_positions`
        "N": x[properties.n_atoms],  # Number of atoms
        "F": x[force_label[dataset]],
    }


def get_generator(base_gen, dataset):
    for x in base_gen:
        yield data_to_dic(x, dataset), x[energy_label[dataset]].float()


def load_data(
    dataset="QM9",
    n_train=100,
    n_test=100,
    batch_size=32,
    molecule="aspirin",
    log=False,
    cache_dir="/home/space/datasets/SchNet",
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
            transforms=[ASENeighborList(cutoff=5.0)],
            split_file=None,
        )
    elif dataset == "MD17":
        valid_molecules = [
            "aspirin",
            "azobenzene",
            "benzene",
            "ethanol",
            "malonaldehyde",
            "naphthalene",
            "paracetamol",
            "salicylic_acid",
            "toluene",
            "uracil",
        ]
        if molecule is None:
            raise Exception("Please specify a molecule")
        elif molecule not in valid_molecules:
            raise Exception(
                f"Please choose one of the following molecules : {valid_molecules}"
            )
        data = MD17(
            os.path.join(cache_dir, "md17.db"),
            # fold = 'reference', # !! new param
            molecule=molecule,
            batch_size=batch_size,
            num_train=n_train,
            num_test=0,
            num_val=n_test,
            transforms=[ASENeighborList(cutoff=5.0)],
            split_file=None,
        )
    elif dataset == "ISO17":
        fix_iso_17_db()
        data = ISO17(
            os.path.join(cache_dir, "iso17.db"),
            fold="reference_eq",
            batch_size=batch_size,
            num_train=n_train,
            num_test=0,
            num_val=n_test,
            transforms=[ASENeighborList(cutoff=5.0)],
            split_file=None,
        )
    else:
        raise ValueError("Only QM9, MD17 and ISO17 are supported but used " + dataset)

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


# class SchnetDataset(Dataset):
#     """Class to load datasets for Schnet."""
#
#     def __init__(self, dataset_iterator, n_train, n_test, molecule="aspirin", fold=None):
#         if dataset == "QM9":
#             data = QM9(
#                 "./qm9.db",
#                 batch_size=5,
#                 num_train=n_train,
#                 num_val=n_test,
#                 transforms=[ASENeighborList(cutoff=5.0)],
#             )
#         elif dataset == "MD17":
#             if molecule is None:
#                 raise Exception("Please specify a molecule")
#             # elif molecule not in ['Aspirin','Azobenzene','Benzene','Ethanol',
#             #                       'Malonaldehyde','Naphthalene','Paracetamol',
#             #                       'Salicylic_acid','Toluene','Uracil']:
#             # raise Exception('Please choose one of the following molecules : aspirin,azobenzene,benzene,ethanol,malonaldehyde,naphthalene,paracetamol,salicylic_acid,toluene,uracil')
#             data = MD17(
#                 "./md17.db",
#                 # fold = 'reference', # !! new param
#                 molecule=molecule,
#                 batch_size=10,
#                 num_train=n_train,
#                 num_val=n_test,
#                 transforms=[ASENeighborList(cutoff=5.0)],
#             )
#         elif dataset == "ISO17":
#             data = ISO17(
#                 "./iso17.db",
#                 fold="reference_eq",
#                 batch_size=10,
#                 num_train=n_train,
#                 num_val=n_test,
#                 transforms=[ASENeighborList(cutoff=5.0)],
#             )
#
#     data.prepare_data()
#     data.setup()
#     test = data.test_dataloader()
#     # val = data.val_dataloader()
#     train = data.train_dataloader()
#     print("Number of reference calculations:", len(data.dataset))
#     print("Number of train data:", len(data.train_dataset))
#     print("Number of validation data:", len(data.val_dataset))
#     print("Number of test data:", len(data.test_dataset))
#     print("Available properties:")
#     for p in data.dataset.available_properties:
#         print("-", p)
#     return get_generator(base_gen=test), get_generator(base_gen=train)
#
#
# def __len__(self):
#     return len(self.landmarks_frame)
#
#
# def __getitem__(self, idx):
#     if torch.is_tensor(idx):
#         idx = idx.tolist()
#
#     img_name = os.path.join(self.root_dir, self.landmarks_frame.iloc[idx, 0])
#     image = io.imread(img_name)
#     landmarks = self.landmarks_frame.iloc[idx, 1:]
#     landmarks = np.array([landmarks], dtype=float).reshape(-1, 2)
#     sample = {"image": image, "landmarks": landmarks}
#
#     if self.transform:
#         sample = self.transform(sample)
#
#     return sample
