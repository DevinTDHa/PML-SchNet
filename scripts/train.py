import torch

from src.baseline import train, BaselineModel

print("CUDA AVAILABLE:", torch.cuda.is_available())
train(BaselineModel(), 'QM9', 50)
