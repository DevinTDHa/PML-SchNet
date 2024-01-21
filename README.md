# PML-SchNet
Project Machine Learning WS23/24: SchNet: A continuous-filter convolutional neural network for modeling quantum interactions


[SchNet: A continuous-filter convolutional neural
network for modeling quantum interactions](https://arxiv.org/pdf/1706.08566.pdf)
[Colab Notebook](https://colab.research.google.com/drive/1h7oTIjv2wdBmQW2EKEvLwJCOmQvYqwGE?usp=sharing)


https://github.com/DevinTDHa/PML-SchNet/assets/33089471/f73b561e-d28c-4848-afc0-f27a2f2f6d39



## Useful commands

### SSH into server
`cd scripts && sh connect.sh`


### Run Training
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


# final run 
pml00@hydra:~/MS2$ srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M all -n 100000 -nt 1000 -b 32"


FAIL RUNS 

    "MD17_energy_and_force_aspirin": {
        "save_path": "model_MD17_energy_and_force_aspirin_20231229_165821.pt",
        "success": false
    },
    "MD17_energy_and_force_azobenzene": {
        "save_path": "model_MD17_energy_and_force_azobenzene_20231229_165821.pt",
        "success": false
    },




-------



## Cross Validate Benchmark Comand 
srun --partition=gpu-test --gpus=1 --pty bash -c "apptainer run --nv ../pml.sif python validation/cross_validation.py" 
