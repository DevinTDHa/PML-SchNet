from src.baseline import *
from src.data_loader import load_data


def test_train_qm9():
    train_qm9(BaselineModelQM9(), "QM9", 5)


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
