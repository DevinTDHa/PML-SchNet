from src.baseline import *
from src.data_loader import load_data

import pytest

@pytest.mark.parametrize("dataset", ["QM9", "MD17", "ISO17"])
def test_train(dataset):
    train(BaselineModel(), dataset, 5)


def test_train_md17():
    train_iter, test_iter = load_data(
        "MD17",
        n_train=1000,
        cache_dir="/home/ducha/Uni-Master/Courses/PML/data",
        log=True,
    )

    model = BaselineModelMD17AspirinEnergy()
    print("Training...")
    train_md17(model, train_iter)
    print("Validation...")
    val_loss = validate_md17(model, test_iter)
    print(val_loss)


def test_train_md17_energy_force():
    train_iter, test_iter = load_data(
        "MD17",
        n_train=1000,
        cache_dir="/home/ducha/Uni-Master/Courses/PML/data",
        log=True,
    )

    model = BaselineModelMD17AspirinEnergyForce()
    print("Training...")
    train_md17_energy_force(model, train_iter)
    print("Validation...")
    val_loss = validate_md17_energy_force(model, test_iter)
    print(val_loss)
