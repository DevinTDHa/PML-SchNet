#!/bin/bash
cd ~/PML-SchNet
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M all --n_ntrain 10000 --n_test 100 -e 1 -lr 0.1"

