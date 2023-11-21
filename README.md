# PML-SchNet
Project Machine Learning WS23/24: SchNet: A continuous-filter convolutional neural network for modeling quantum interactions


[SchNet: A continuous-filter convolutional neural
network for modeling quantum interactions](https://arxiv.org/pdf/1706.08566.pdf)
[Colab Notebook](https://colab.research.google.com/drive/1h7oTIjv2wdBmQW2EKEvLwJCOmQvYqwGE?usp=sharing)


https://github.com/DevinTDHa/PML-SchNet/assets/33089471/f73b561e-d28c-4848-afc0-f27a2f2f6d39
https://tubcloud.tu-berlin.de/s/nekRGHF25Wt5Wko


## Useful commands

### SSH into server
`sh scripts/cluster/connect.sh`

### Get a GPU Shell 
`srun --partition=gpu-test --gpus=1 --pty bash   # Connect to GPU shell`


```bash
# 1) 
cd ~/PML-SchNet

# 2) Start a Jo session
srun --partition=gpu-test --gpus=1 --pty bash   

# 3.1) run training with force+energy prediction when applicable
apptainer run --nv pml.sif python scripts/train.py -M md17_one_molecule
apptainer run --nv pml.sif python scripts/train.py -M md17_all_molecules
apptainer run --nv pml.sif python scripts/train.py -M iso17
apptainer run --nv pml.sif python scripts/train.py -M qm9
apptainer run --nv pml.sif python scripts/train.py -M all
apptainer run --nv pml.sif python scripts/train.py -M all_one_molecule


# 3.2) Or customize your own training
apptainer run --nv pml.sif python scripts/train.py -d MD17 -m aspirin -t energy


```
