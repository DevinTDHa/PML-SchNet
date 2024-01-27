import sys

sys.path.append(".")

from pml_schnet.data_loader import load_data

n_train = 10_000
n_test = 1_000

batch_size = 32
epochs = 100
lr = 1e-3

train_set, test_set = load_data(
    "ISO17",
    n_train,
    n_test,
    batch_size=batch_size,
    split_file=None,
    molecule="NA",
    keep_in_memory=True,
    cache_pickle=True,
)

print(len(train_set), len(test_set))
