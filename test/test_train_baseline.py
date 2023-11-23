import pytest

from pml_schnet.baseline import *
from pml_schnet.settings import (
    Trainable,
    all_trainable_one_molecule,
)


# all_trainable
def test_id(test: Trainable) -> str:
    return f"{test.dataset}_{test.task}_{test.molecule}"


# @pytest.mark.parametrize("trainable", md17_trainable_one_molecule, ids=test_id)
# @pytest.mark.parametrize("trainable", qm9_trainable, ids=test_id)
@pytest.mark.parametrize("trainable", all_trainable_one_molecule, ids=test_id)
def test_train(trainable: Trainable):
    train_and_validate(trainable, "baseline")


def test_train_iso17_energy_force():
    train_params = Trainable(Dataset.iso17, Task.force)
    train_and_validate(train_params, Model.baseline, epochs=1)
