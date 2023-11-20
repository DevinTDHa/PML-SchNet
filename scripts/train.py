import torch
import argparse
import sys
import os
sys.path.append(os.getcwd())
from src.baseline import train, BaselineModel

# Initialize parser
parser = argparse.ArgumentParser(description="Train a baseline model")

# Adding optional arguments
parser.add_argument("-d", "--dataset", type=str, default="QM9", help="Dataset to use (QM9, MD17, ISO17)")
parser.add_argument("-e", "--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="Learning rate")

# Read arguments from command line
args = parser.parse_args()

print("CUDA AVAILABLE:", torch.cuda.is_available())

# Call the train function with CLI arguments
train(BaselineModel(), dataset=args.dataset, epochs=args.epochs, lr=args.learning_rate)



# apptainer run --nv pml.sif python scripts/train.py