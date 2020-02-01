"""
Models using a neural net and standard precomputed features.
"""
import torch
from torch import nn
from torch.utils.data import TensorDataset


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        
