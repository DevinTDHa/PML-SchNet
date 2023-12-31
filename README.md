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
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M iso17"
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M qm9"
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M all"
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -M all_one_molecule"

# 2.2) Or customize your own training
srun --partition=gpu-2d --gpus=1 --pty bash -c "apptainer run --nv pml.sif python scripts/train.py -d MD17 -m paracetamol -t energy"

```


