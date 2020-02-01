import sys
import os

import numpy as np
import torch
from torch import optim
import tqdm

from models.deep_stdfeatures import Net
from utils.loader import MyPointCloud

from torch.utils.data import DataLoader


def train(model: Net, optim: torch.optim.Optimizer, dataset: MyPointCloud, num_epochs: int):
    criterion = torch.nn.NLLLoss()
    
    for epoch in tqdm.trange(num_epochs):
        points, labels, features = dataset.get_sample()
        
        # get feature vector
        x_ = torch.cat(features, dim=1)
        output = model(x_)
        loss = criterion(output, labels)
        loss.backward()
        # import ipdb; ipdb.set_trace()
        optim.step()
        optim.zero_grad()
        
        prediction = np.argmax(output.data.numpy(), axis=1)
        accuracy = np.mean(prediction == labels.data.numpy())
        
        print("Epoch %d. Current loss: %.3e. Accuracy: %.2f"
              % (epoch,loss.data.numpy(),100*accuracy))


NUM_EPOCHS = 50

if __name__ == "__main__":
    dataset = MyPointCloud("data/MiniChallenge/training/MiniLille1.ply")
    # import ipdb; ipdb.set_trace()
    model = Net(9)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, optimizer, dataset, NUM_EPOCHS)
    
    
    
    
    

