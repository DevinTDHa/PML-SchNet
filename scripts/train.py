import argparse
import os
import sys
from pprint import pprint

import torch

sys.path.append(os.getcwd())
from src.settings import train_modes, Trainable
from src.baseline import train_and_validate

parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("-d", "--dataset", type=str, default="QM9", help="Dataset to use (QM9, MD17, ISO17)")
parser.add_argument("-m", "--molecule", type=str, default="aspirin", help="Molecule to use for MD17 dataset")
parser.add_argument("-t", "--task", type=str, default="energy", help="energy or force prediction task")
parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("-n", "--n_train", type=int, default=100, help="number of training samples")
parser.add_argument("-nt", "--n_test", type=int, default=100, help="number of test samples")
parser.add_argument("-M", "--train_mode", type=str, default=None,
                    help=f"Pre-Configured training bundles, always force+energy if applicable. One of {train_modes.keys()}")
args = parser.parse_args()
print("ARGS ARE :", args)

print("CUDA AVAILABLE:", torch.cuda.is_available())

if args.train_mode:
    if args.train_mode not in train_modes.keys():
        raise ValueError(f"train_mode must be one of {train_modes.keys()}")
    else:
        results = {}
        for trainable in train_modes[args.train_mode]:
            results[str(trainable)] = {}
            print(f"Training {trainable}")
            try:
                print("Training...")
                train_loss, test_loss = train_and_validate(trainable, 'baseline', epochs=args.epochs,
                                                          n_train=args.n_train,
                                                          n_test=args.n_test,
                                                          lr=args.learning_rate)
                results[str(trainable)]['success'] = True
                results[str(trainable)]['train_loss'] = True
                results[str(trainable)]['test_loss'] = True
            except Exception as e:
                results[str(trainable)]['success'] = False
                import traceback

                traceback.print_exc()

                print(f"Error {e} while training {trainable}")
                continue
        print("Done!")
        pprint(results)

else:
    trainable = Trainable(
        dataset=args.dataset,
        task=args.task,
        molecule=args.molecule
    )
    train_and_validate(trainable, 'baseline', epochs=args.epochs, n_train=args.n_train, lr=args.learning_rate,
                       n_test=args.n_test,
                       )

# TODO maybe broken ./iso17.db/iso17/reference_eq.db:
# Todo make sure we always take max datasize??