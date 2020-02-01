"""
Models using a neural net and standard precomputed features.
"""
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import TensorDataset


class Net(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        
        self.lin1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.lin2 = nn.Linear(hidden_size, 7)
        self.out = nn.LogSoftmax(dim=1)  # this works with NLLLoss
        

    def forward(self, x: Tensor):
        x = self.lin1(x)
        x = self.lin2(x)
        y = self.out(x)
        return y

