import os.path
import shutil

from torch import nn

from pml_schnet.data_loader import load_data
from pml_schnet.model import BaselineModel, SchNet
from pml_schnet.settings import Model, Task, Trainable
from pml_schnet.settings import device
from pml_schnet.training import (
    train_baseline_energy,
    train_baseline_force,
    train_baseline_energy_force,
    train_schnet_energy,
    validate_schnet_energy,
    validate_schnet_force_energy,
    train_schnet_energy_force,
    train_schnet_force,
    validate_schnet_force,
)
from pml_schnet.validation import (
    validate_baseline_energy,
    validate_baseline_force,
    validate_baseline_energy_force,
)


# High level logic routing


def train(
    model,
    dataset,
    task,
    molecule,
    epochs,
    lr,
    n_train,
    n_test,
    batch_size,
    split_file,
):
    # generic train router for all models
    if molecule is not None and dataset != "MD17":
        raise ValueError("Molecule can only be specified for MD17 dataset")
    model_obj = get_model(model)
    model_obj = model_obj.to(device)

    if os.path.exists(split_file + ".npz"):
        print("Old splitfile detected, removing...")
        shutil.rmtree(split_file + ".npz", ignore_errors=True)

    if model == Model.baseline:
        if task == Task.force:
            return model_obj, train_baseline_force(
                model_obj, n_train, n_test, lr, epochs, dataset
            )

        elif task == Task.energy_and_force:
            return model_obj, train_baseline_energy_force(
                model_obj, n_train, n_test, lr, epochs, dataset
            )
        elif task == Task.energy:
            return model_obj, train_baseline_energy(
                model_obj, n_train, n_test, lr, epochs, dataset
            )
    elif model == Model.schnet:
        if task == Task.force:
            train_losses, val_losses = train_schnet_force(
                model_obj,
                n_train,
                n_test,
                lr,
                epochs,
                dataset,
                batch_size,
                split_file=split_file,
                molecule=molecule,
            )
            return model_obj, train_losses, val_losses
        elif task == Task.energy_and_force:
            train_losses, val_losses = train_schnet_energy_force(
                model_obj,
                n_train,
                n_test,
                lr,
                epochs,
                dataset,
                batch_size,
                split_file=split_file,
                molecule=molecule,
            )
            return model_obj, train_losses, val_losses
        elif task == Task.energy:
            train_losses, val_losses = train_schnet_energy(
                model_obj,
                n_train,
                n_test,
                lr,
                epochs,
                dataset,
                batch_size,
                split_file=split_file,
                molecule=molecule,
            )
            return model_obj, train_losses, val_losses

    raise ValueError("Invalid Task or Dataset, could not train model")


def train_and_validate(
    trainable: Trainable,
    n_train,
    n_test,
    lr,
    epochs,
    model,
    return_model,
    batch_size,
    return_labels_for_test_only,
    model_save_name,
    split_save_folder,
):
    # model_type = model

    os.makedirs(split_save_folder, exist_ok=True)
    split_file_name = split_save_folder + model_save_name + "_split"

    model, train_losses, val_losses = train(
        model=model,
        dataset=trainable.dataset,
        task=trainable.task,
        molecule=trainable.molecule,
        epochs=epochs,
        n_train=n_train,
        n_test=n_test,
        lr=lr,
        batch_size=batch_size,
        split_file=split_file_name,
    )

    # test_loss, predicted_labels = validate(
    #     model,
    #     model_type,
    #     trainable.dataset,
    #     trainable.task,
    #     n_train=n_train,
    #     n_test=n_test,
    #     molecule=trainable.molecule,
    #     split_file=split_file_name,
    # )

    if return_model:
        # TODO maybe smarter
        model.writer = None
        return train_losses, val_losses, model
    # elif return_labels_for_test_only:
    #     return predicted_labels
    else:
        return train_losses, val_losses


def validate(
    model,
    model_type,
    dataset,
    task,
    molecule,
    n_train,
    n_test,
    split_file,
):
    if model_type == Model.baseline:
        if task == Task.energy:
            return validate_baseline_energy(model, dataset, n_train, n_test, molecule)
        elif task == Task.force:
            return validate_baseline_force(model, dataset, n_train, n_test, molecule)
        elif task == Task.energy_and_force:
            return validate_baseline_energy_force(
                model, dataset, n_train, n_test, molecule
            )
    elif model_type == Model.schnet:
        _, test_gen = load_data(
            dataset,
            n_train=n_train,
            n_test=n_test,
            molecule=molecule,
            split_file=split_file,
        )
        if task == Task.energy:
            return validate_schnet_energy(model, test_gen, nn.L1Loss())
        elif task == Task.force:
            return validate_schnet_force(model, test_gen, nn.L1Loss())
        elif task == Task.energy_and_force:
            return validate_schnet_force_energy(model, test_gen)
    else:
        print()
        raise ValueError(
            f"Invalid Task or Dataset, could not validate model "
            f"for {dataset, task, molecule, n_train, n_test}"
        )


# def train_apply(
#     method,  #
#     dataset,  # QM9, MD17, ISO17
#     task,  # energy, force, energy_and_force
#     molecule,  # ...
#     n_train,
#     n_test,
#     lr,
#     epochs,
#     batch_size,
# ):
#     """
#
#     :param method:  baseline or schnet
#     :param dataset: QM9, MD17, ISO17
#     :param task: energy, force, energy_and_force
#     :param molecule: aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene, paracetamol, salicylic_acid, toluene or  uracil
#     :param n_train: num of train samples
#     :param n_test: num of test samples
#     :param lr: learning rate
#     :param epochs: epochs
#     :param save_path: where to store model, if None model not saved
#     :param batch_size: batchsize during training
#     :return: predicted labels for test data
#     """
#     return train_and_validate(
#         Trainable(dataset, molecule, task),
#         model=method,
#         n_train=n_train,
#         n_test=n_test,
#         lr=lr,
#         epochs=epochs,
#         batch_size=batch_size,
#         return_labels_for_test_only=True,
#     )


def get_model(model, writer=None):
    if model == Model.baseline:
        return BaselineModel()
    elif model == Model.schnet:
        return SchNet(writer=writer)
