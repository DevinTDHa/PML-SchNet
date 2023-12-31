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
    validate_schnet,
    validate_schnet_force_energy,
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
    molecule=None,
    epochs=1,
    lr=0.01,
    n_train=100,
    n_test=100,
    batch_size=32,
    writer=None,
):
    # generic train router for all models
    if molecule is not None and dataset != "MD17":
        raise ValueError("Molecule can only be specified for MD17 dataset")
    model_obj = get_model(model, writer)
    model_obj = model_obj.to(device)

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
        # TODO train_schnet_force or train_schnet_energy enough?
        if task == Task.force:
            return model_obj, train_schnet_energy(
                model_obj, n_train, n_test, lr, epochs, dataset, batch_size
            )

        elif task == Task.energy_and_force:
            return model_obj, train_schnet_energy(
                model_obj, n_train, n_test, lr, epochs, dataset, batch_size
            )
        elif task == Task.energy:
            return model_obj, train_schnet_energy(
                model_obj,
                n_train,
                n_test,
                lr,
                epochs,
                dataset,
                batch_size,
            )

    raise ValueError("Invalid Task or Dataset, could not train model")


def validate(model, model_type, dataset, task, molecule, n_train, n_test):
    # returns loss,predicted_labels
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
        train_gen, test_gen = load_data(
            dataset, n_train=n_train, n_test=n_test, molecule=molecule
        )

        if task == Task.energy:
            return validate_schnet(model, test_gen, nn.L1Loss())
        elif task == Task.force:
            return validate_schnet(model, test_gen, nn.L1Loss())
        elif task == Task.energy_and_force:
            return validate_schnet_force_energy(model, test_gen)
    else:
        print()
        raise ValueError(
            f"Invalid Task or Dataset, could not validate model "
            f"for {dataset, task, molecule, n_train, n_test}"
        )


def train_and_validate(
    trainable: Trainable,
    model="schnet",
    n_train=10,
    n_test=10,
    lr=0.01,
    epochs=2,
    return_model=None,
    batch_size=32,
    writer=None,
    return_labels_for_test_only=False,
):
    print("Training...")
    model_type = model
    model, train_loss = train(
        model=model,
        dataset=trainable.dataset,
        task=trainable.task,
        molecule=trainable.molecule,
        epochs=epochs,
        n_train=n_train,
        n_test=n_test,
        lr=lr,
        batch_size=batch_size,
        writer=writer,
    )
    print("Training loss : ", train_loss)
    test_loss, predicted_labels = validate(
        model,
        model_type,
        trainable.dataset,
        trainable.task,
        n_train=n_train,
        n_test=n_test,
        molecule=trainable.molecule,
    )
    print("Test loss : ", test_loss)

    if return_model:
        # TODO maybe smarter
        model.writer = None
        return train_loss, test_loss, model

    if return_labels_for_test_only:
        return predicted_labels
    return train_loss, test_loss


def train_apply(
    method="method_name",  #
    dataset="dataset_name",  # QM9, MD17, ISO17
    task="energy",  # energy, force, energy_and_force
    molecule="aspirin",  # ...
    n_train=10,
    n_test=10,
    lr=0.01,
    epochs=2,
    save_path=None,
    batch_size=32,
):
    """

    :param method:  baseline or schnet
    :param dataset: QM9, MD17, ISO17
    :param task: energy, force, energy_and_force
    :param molecule: aspirin, azobenzene, benzene, ethanol, malonaldehyde, naphthalene, paracetamol, salicylic_acid, toluene or  uracil
    :param n_train: num of train samples
    :param n_test: num of test samples
    :param lr: learning rate
    :param epochs: epochs
    :param save_path: where to store model, if None model not saved
    :param batch_size: batchsize during training
    :return: predicted labels for test data
    """
    return train_and_validate(
        Trainable(dataset, molecule, task),
        model=method,
        n_train=n_train,
        n_test=n_test,
        lr=lr,
        epochs=epochs,
        save_path=save_path,
        batch_size=batch_size,
        return_labels_for_test_only=True,
    )


def get_model(model, writer=None):
    if model == Model.baseline:
        return BaselineModel()
    elif model == Model.schnet:
        return SchNet(writer=writer)
