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
        
        self.batchnorm = nn.BatchNorm1d(3, track_running_stats=False, affine=False)
        self.lin1 = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
        )
        
        self.linglob = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.ReLU()
        )
        
        self.out = nn.Sequential(
            nn.Linear(hidden_size+128, OUT_SIZE),
            nn.LogSoftmax(dim=1)  # this works with NLLLoss
        )

    def forward(self, points: Tensor, x: Tensor):
        n_pts = points.shape[0]
        points = self.batchnorm(points)
        z = torch.cat((points, x), 1)
        z = self.lin1(z)
        globfeat = self.linglob(z)
        globfeat = torch.mean(globfeat, dim=0, keepdim=True)  # (1,128)
        z = torch.cat((z, globfeat.repeat(n_pts, 1)), 1)  # (n_pts,H+128)
        output = self.out(z)
        return output

