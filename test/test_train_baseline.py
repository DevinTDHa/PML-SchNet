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
    print("Training...")
    n_train = 100
    model, test_losses = train(model='baseline', dataset=trainable.dataset, task=trainable.task,
                               molecule=trainable.molecule,
                               epochs=2, n_train=n_train)
    if trainable.dataset == Dataset.md17:
        #     TODO validation for other models
        print("Validation...")
        val_loss = validate(model, trainable.dataset, trainable.task, n_train=n_train, molecule=trainable.molecule)
        print(val_loss)
