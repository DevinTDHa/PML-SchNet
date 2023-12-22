import argparse
import os
import pickle
import shutil

import numpy as np
import torch.nn
from sklearn.model_selection import KFold
from tqdm import tqdm

from pml_schnet.data_loader import load_data
from pml_schnet.loss import *
from pml_schnet.model import SchNet
from pml_schnet.settings import *


class CrossValidator:
    def __init__(self, batch_size, n_train, n_test, epochs, lr, k):
        self.batch_size = batch_size
        self.n_train = n_train
        self.n_test = n_test
        self.epochs = epochs
        self.lr = lr
        self.k = k

        split_file = "cross_validation_split.npz"
        if os.path.exists(split_file):
            os.remove(split_file)

        train_gen, test_gen = load_data(
            "ISO17",
            n_train=n_train,
            n_test=n_test,
            batch_size=batch_size,
            split_file=split_file,
        )
        print("Reading data...")
        self.data_batches = list(train_gen)
        self.val_batches = list(test_gen)

        with open("val_set.pickle", "wb") as f:
            pickle.dump(self.val_batches, f)

        kf = KFold(n_splits=k)
        self.kfolds = list(kf.split(self.data_batches))

        self.runs_folder = "kf_runs"
        shutil.rmtree(self.runs_folder, ignore_errors=True)
        os.makedirs(self.runs_folder, exist_ok=True)

    @staticmethod
    def save_model(model, name):
        torch.save(model.state_dict(), f"{name}.pt")

    @staticmethod
    def load_model(path):
        model = SchNet()
        model.load_state_dict(torch.load(path))
        model.to(device)
        return model

    def train_split(self, train_indexes, test_indexes, split_i):
        model = SchNet().to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        losses = []

        with tqdm(total=self.epochs, ncols=80, position=1) as progress_bar:
            progress_bar.set_description(f"Split {split_i}")
            for epoch in range(self.epochs):
                loss = None
                for train_batch in train_indexes:
                    X_batch, y_batch = self.data_batches[train_batch]
                    # Forward pass
                    X_batch["R"].requires_grad_()
                    F = X_batch["F"].to(device)

                    # Forward pass
                    E_pred = model(X_batch)
                    loss = energy_force_loss(
                        E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F
                    )
                    # Backward pass and optimization
                    optimizer.zero_grad()  # Clear gradients
                    loss.backward()  # Compute gradients

                    optimizer.step()  # Update weights

                    progress_bar.set_postfix(train_loss=f"{loss:.4E}", refresh=False)

                progress_bar.update(1)
                losses.append(loss.item())

        print(f"Split {split_i} Last Train Loss:", losses[-1])

        # Test Splits For Training data
        model.eval()

        test_losses = []
        for test_batch in test_indexes:
            X_batch, y_batch = self.data_batches[test_batch]

            # Forward pass
            X_batch["R"].requires_grad_()
            F = X_batch["F"].to(device)

            # Forward pass
            E_pred = model(X_batch)
            loss = energy_force_loss(E_pred=E_pred, R=X_batch["R"], E=y_batch, F=F)

            test_losses.append(loss.item())

        print(f"Split {split_i} Mean Test Error:", np.array(test_losses).mean())

        # Saving Everything
        model_name = f"{self.runs_folder}/model_split_{split_i}"
        self.save_model(model, model_name)
        np.save(model_name + "_train_losses.npy", losses)
        np.save(model_name + "_test_losses.npy", test_losses)

    def cross_validate(self):
        for i, fold in tqdm(enumerate(self.kfolds), desc="K", total=self.k, position=0):
            train_indexes, test_indexes = fold
            self.train_split(train_indexes, test_indexes, i)


def create_arg_parser():
    def positive_int(value):
        ivalue = int(value)
        if ivalue <= 0:
            raise argparse.ArgumentTypeError("%s is not a positive integer" % value)
        return ivalue

    def positive_float(value):
        fvalue = float(value)
        if fvalue <= 0:
            raise argparse.ArgumentTypeError("%s is not a positive float" % value)
        return fvalue

    parser = argparse.ArgumentParser(description="Cross Validation Parameters")

    parser.add_argument(
        "--batch_size",
        type=positive_int,
        default=32,
        help="Batch size for training (default: 32)",
    )

    parser.add_argument(
        "--n_train",
        type=positive_int,
        default=100000,
        help="Number of training samples (default: 100000)",
    )

    parser.add_argument(
        "--n_test",
        type=positive_int,
        default=None,
        help="Number of testing samples (default: n_train * 0.2)",
    )

    parser.add_argument(
        "--k",
        type=positive_int,
        default=5,
        help="Number of folds for cross-validation (default: 5)",
    )

    parser.add_argument(
        "--epochs",
        type=positive_int,
        default=2,
        help="Number of epochs for training (default: 1)",
    )

    parser.add_argument(
        "--lr",
        type=positive_float,
        default=0.01,
        help="Learning rate for training (default: 0.01)",
    )

    return parser


if __name__ == "__main__":
    parser = create_arg_parser()
    args = parser.parse_args()

    # If n_test is not provided, set it to 20% of n_train
    if args.n_test is None:
        args.n_test = int(args.n_train * 0.2)

    # Display the parsed arguments
    print("Cross Validation. Parameters:")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Training Samples: {args.n_train}")
    print(f"Number of Testing Samples: {args.n_test}")
    print(f"Number of Folds (k): {args.k}")
    print(f"Number of Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")

    # Get Data
    cross_validator = CrossValidator(
        args.batch_size, args.n_train, args.n_test, args.epochs, args.lr, args.k
    )
    cross_validator.cross_validate()
