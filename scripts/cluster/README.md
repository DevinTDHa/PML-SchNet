# Useful Cluster Commands

## SSH into server

`cd scripts && sh connect.sh`

## Run Training

Launches job once SLURM session is available.

```shell
# 1)
cd ~/PML-SchNet


# alternative partitions cpu-test, gpu-test, etc..
# 2.1) run training with force+energy prediction when applicable
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M md17_one_molecule"
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M md17_all_molecules"

srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M qm9"
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M iso17"

srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M all"
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M all_one_molecule"


srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M qm9 --n_train=100000 --n_test=30000 -e 1"
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M iso17 --n_train=100000 --n_test=30000 -e 1"

# 2.2) Or customize your own training
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -d MD17 -m paracetamol -t energy_and_force"

# --> 1) Get all metrics + save model , make table in paper, write some shit

# main runner  FOR MS 2
time srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M qm9 --n_train 10000 --n_test 1000 -e 1 -lr 0.1 --save True"


# Quick MS2 Tests 1 Per Dataset
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M all --n_train 50000 --n_test 1000 -e 1 -lr 0.1"

srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M all --n_train 10000 --n_test 1000 -e 1 -lr 0.1"

# Nothing todo run
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M all --n_train 200000 --n_test 2000 -e 10 -lr 0.1"

# QUick mini test
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M all --n_train 30 --n_test 10 -e 1 -lr 0.1"
# ALL MOLECLE RUN # model_test_run_20231229_165821.json
# ALLES ANDERE
ntest = 100
epochs = 1
lr = 0.1


# Paper write commands
## Train one specific molecule either energy/force or energy_and_force for S size
time srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -d MD17 --n_train 10000 --n_test 1000 -e 1 -lr 0.1 -m azobenzene -t energy_and_force"
time srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -d qm9 --n_train 10000 --n_test 1000 -e 1 -lr 0.1 -t energy_and_force"
time srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -d iso17 --n_train 10000 --n_test 1000 -e 1 -lr 0.1 -t energy_and_force"

# Re-Train all models for M size
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M all --n_train 300000 --n_test 10000 -e 2 -lr 0.1"

srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M iso17 --n_train 300000 --n_test 10000 -e 2 -lr 0.1"
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M qm9 --n_train 300000 --n_test 10000 -e 2 -lr 0.1"


# QM9 --> N 50k, 100k, 110462K
# MD17 --> 1k, 50k
# ISO17 --> 5000 train, 5000 test
# srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M iso17 --n_train 50000 --n_test 10000 -e 2 -lr 0.01"
# NEW MAIN TRAIN PARAMS
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M iso17 --n_train 50000 --n_test 10000 -e 2 -lr 0.01"


lrs = 0.1
epochs = 1
```

## Cross Validate Benchmark Command

```bash
srun --partition=gpu-test --gpus=1 --pty bash -c "apptainer run --nv ../pml.sif python validation/cross_validation.py"
```

## GPU Shell

Run tmux, so we get a detachable shell

```bash
tmux
```

Request a shell, that resides on a partition with access to an NVIDIA GPU.

```bash
srun --partition=gpu-2d --gpus=1 --pty bash
```

Run the python script with an NVIDIA GPU, using a prebuilt docker container (here `ms3.sif`).

```bash
apptainer run --nv ms3.sif python scripts/regularization/train_reg.py
```

(Optional) Save tmux outputs after detaching with `ctrl+b, d`

```bash
tmux capture-pane -pS - > ~/tmux-buffer.txt
```
