import argparse
import json
import os
import sys
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())

from pml_schnet.route import train_and_validate
from pml_schnet.settings import train_modes, Trainable

# TODO's
# Log duration/paramerize method/
parser = argparse.ArgumentParser(description="Train a model")
parser.add_argument(
    "-d", "--dataset", type=str, default="QM9", help="Dataset to use (QM9, MD17, ISO17)"
)
parser.add_argument(
    "-m",
    "--molecule",
    type=str,
    default="aspirin",
    help="Molecullse to use for MD17 dataset",
)
parser.add_argument(
    "-t", "--task", type=str, default="energy", help="energy or force prediction task"
)
parser.add_argument("-e", "--epochs", type=int, default=1, help="Number of epochs")
parser.add_argument(
    "-lr", "--learning_rate", type=float, default=0.01, help="Learning rate"
)
parser.add_argument(
    "-n", "--n_train", type=int, default=1, help="number of training samples"
)
parser.add_argument(
    "-nt", "--n_test", type=int, default=1, help="number of test samples"
)
parser.add_argument("-b", "--batch_size", default=1, type=int, help="batchsize ")

parser.add_argument("-s", "--save", type=bool, default=True, help="save final model")

parser.add_argument(
    "-M",
    "--train_mode",
    type=str,
    default=None,
    help=f"Pre-Configured training bundles, always force+energy if applicable. One of {train_modes.keys()}",
)
args = parser.parse_args()
print("ARGS ARE :", args)

print("CUDA AVAILABLE:", torch.cuda.is_available())
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_folder = f"runs/{args.train_mode}_{timestamp}"
os.makedirs(save_folder, exist_ok=True)

if args.train_mode:
    if args.train_mode not in train_modes.keys():
        raise ValueError(f"train_mode must be one of {train_modes.keys()}")
    else:
        results = {}
        for trainable in train_modes[args.train_mode]:
            results[str(trainable)] = {}
            print(f"Training {trainable}...")
            try:
                model_name = f"model_{trainable}_{timestamp}"
                model_folder = f"{save_folder}/{model_name}/"
                writer = SummaryWriter(model_folder)
                save_path = f"{model_folder}/{model_name}.pt" if args.save else None
                results[str(trainable)]["save_path"] = save_path
                print(f"Training {model_name}")
                train_loss, test_loss, model = train_and_validate(
                    trainable=trainable,
                    model="schnet",
                    epochs=args.epochs,
                    n_train=args.n_train,
                    n_test=args.n_test,
                    lr=args.learning_rate,
                    return_model=True if save_path else False,
                    batch_size=args.batch_size,
                    writer=writer,
                    return_labels_for_test_only=False,
                    model_save_name=model_name,
                )
                results[str(trainable)]["success"] = True
                results[str(trainable)]["train_loss"] = train_loss[-1]
                results[str(trainable)]["test_loss"] = np.mean(test_loss)
                print("Saving model to", save_path)
                torch.save(model, save_path)
                np.savetxt(f"{model_folder}/{model_name}_train_loss.txt", train_loss)
                np.savetxt(f"{model_folder}/{model_name}_test_loss.txt", test_loss)
            except Exception as e:
                results[str(trainable)]["success"] = False
                import traceback

                traceback.print_exc()
                print(f"Error {e} while training {trainable}")
                continue
        print("Done!")
        pprint(results)

        summary_file = f"{save_folder}/summary_model_test_run_{timestamp}.json"
        with open(summary_file, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Training Summary dumped to {summary_file}")

else:
    raise NotImplementedError()
    # trainable = Trainable(dataset=args.dataset, task=args.task, molecule=args.molecule)
    # tf_logs_dir = f"runs/model_{trainable}_{timestamp}_logs"
    # writer = SummaryWriter(tf_logs_dir)
    # save_path = f"runs/model_{trainable}_{timestamp}.pt" if args.save else None
    # train_and_validate(
    #     trainable,
    #     model="schnet",
    #     epochs=args.epochs,
    #     n_train=args.n_train,
    #     lr=args.learning_rate,
    #     n_test=args.n_test,
    #     batch_size=args.batch_size,
    #     writer=writer,
    # )


# Data dumped to model_test_run_20231229_165821.json 100min
