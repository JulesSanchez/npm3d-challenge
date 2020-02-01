"""
Models using a neural net and standard precomputed features.
"""
import torch
from torch import Tensor
from torch import nn
from torch.utils.data import TensorDataset


class Net(nn.Module):
    """A simple two-layer neural network model.
    
    Feed it the pre-computed features."""
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        OUT_SIZE = 7
        
        self.batchnorm = nn.BatchNorm1d(3)
        self.lin1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU()
        )
        self.lin2 = nn.Linear(hidden_size, OUT_SIZE)
        self.out = nn.LogSoftmax(dim=1)  # this works with NLLLoss
        

    def forward(self, points: Tensor, x: Tensor):
        points = self.batchnorm(points)
        z = torch.cat((points, x), 1)
        z = self.lin1(z)
        z = self.lin2(z)
        y = self.out(z)
        return y

