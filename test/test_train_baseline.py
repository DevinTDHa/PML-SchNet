import pytest

from src.baseline import *
from src.settings import Trainable, md17_trainable_one_molecule, all_trainable_one_molecule, qm9_trainable


# all_trainable
def test_id(test: Trainable) -> str:
    return f"{test.dataset}_{test.task}_{test.molecule}"


# @pytest.mark.parametrize("trainable", md17_trainable_one_molecule, ids=test_id)
# @pytest.mark.parametrize("trainable", qm9_trainable, ids=test_id)
@pytest.mark.parametrize("trainable", all_trainable_one_molecule, ids=test_id)
def test_train(trainable: Trainable):
    train_and_validate(trainable,'baseline')
