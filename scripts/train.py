import argparse
import os
import sys

import torch


sys.path.append(os.getcwd())
from src.settings import train_modes
from src.baseline import train

parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument("-d", "--dataset", type=str, default="QM9", help="Dataset to use (QM9, MD17, ISO17)")
parser.add_argument("-m", "--molecule", type=str, default="aspirin", help="Molecule to use for MD17 dataset")
parser.add_argument("-t", "--task", type=str, default="energy", help="energy or force prediction task")
parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("-n", "--n_train", type=int, default=100, help="number of training samples")
parser.add_argument("-M", "--train_mode", type=str, default=None,
                    help=f"Pre-Configured training bundles, always force+energy if applicable. One of {train_modes.keys()}")
args = parser.parse_args()

print("CUDA AVAILABLE:", torch.cuda.is_available())

if args.train_mode:
    if args.train_mode not in train_modes.keys():
        raise ValueError(f"train_mode must be one of {train_modes.keys()}")
    else:
        for trainable in train_modes[args.train_mode]:
            print(f"Training {trainable}")
            try:
                train(model='baseline', dataset=trainable.dataset, task=trainable.task,
                      molecule=trainable.molecule,
                      epochs=args.epochs, n_train=args.n_train)

            except Exception as e:
                print(f"Error {e} while training {trainable}")
                continue
else:
    train(model='baseline', dataset=args.dataset, task=args.task,
          molecule=args.molecule,
          epochs=args.epochs, n_train=args.n_train)

"""
How to use :
1. SSH into Cluster
2. cd ~/PML-SchNet
3. srun --partition=gpu-test --gpus=1 --pty bash   # Connect to GPU shell 
4. apptainer run --nv pml.sif python scripts/train.py -M md17_one_molecule

"""
# apptainer run --nv pml.sif python scripts/train.py

