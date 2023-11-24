from pml_schnet.model import BaselineModel
from pml_schnet.settings import Model, Task, Trainable
from pml_schnet.settings import device
from pml_schnet.training import (
    train_baseline_energy,
    train_baseline_force,
    train_baseline_energy_force,
)
from pml_schnet.validation import (
    validate_baseline_energy,
    validate_baseline_force,
    validate_baseline_energy_force,
)
# High level logic routing

def train(
    model, dataset, task, molecule=None, epochs=1, lr=0.01, n_train=100, n_test=100
):
    # generic train router for all models
    if molecule is not None and dataset != "MD17":
        raise ValueError("Molecule can only be specified for MD17 dataset")
    model_obj = get_model(model, dataset, task, molecule)
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
    raise ValueError("Invalid Task or Dataset, could not train model")


def validate(model, dataset, task, molecule, n_train, n_test):
    if task == Task.energy:
        return validate_baseline_energy(model, dataset, n_train, n_test, molecule)
    elif task == Task.force:
        return validate_baseline_force(model, dataset, n_train, n_test, molecule)
    elif task == Task.energy_and_force:
        return validate_baseline_energy_force(model, dataset, n_train, n_test, molecule)


def train_and_validate(
    trainable: Trainable, model="baseline", n_train=10, n_test=10, lr=0.2, epochs=2
):
    print("Training...")
    model, train_loss = train(
        model=model,
        dataset=trainable.dataset,
        task=trainable.task,
        molecule=trainable.molecule,
        epochs=epochs,
        n_train=n_train,
        n_test=n_test,
        lr=lr,
    )
    print("Training loss : ", train_loss)
    test_loss = validate(
        model,
        trainable.dataset,
        trainable.task,
        n_train=n_train,
        n_test=n_test,
        molecule=trainable.molecule,
    )
    print("Test loss : ", test_loss)

    return train_loss, test_loss


def get_model(model, dataset, task, molecule="aspirin"):
    if model == Model.baseline:
        return BaselineModel()
        # if dataset in [Dataset.qm9, Dataset.iso17]:
        #     return BaselineModel()
        # elif dataset == Dataset.md17:
        #     if molecule is None:
        #         raise ValueError("Please specify a molecule for md17")
        #     if task == "energy":
        #         return BaselineModelMD17AspirinEnergy(molecule)
        #     elif task == "force":
        #         return BaselineModelMD17AspirinEnergyForce(molecule)
        #     else:
        #         raise ValueError("Invalid Task or Dataset, could not load model")
    elif model == Model.schnet:
        raise NotImplemented("Schnet not implemented yet")
    else:
        raise ValueError("Not supported model", model)
