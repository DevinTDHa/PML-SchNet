from src.baseline import train, BaselineModel
from src.data_loader import load_data


def test_train():
    test_gen, train_gen = load_data('QM9', 100, 100, )
    train(BaselineModel(), train_gen, 5)
