import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.deep_stdfeatures import Net

import tqdm
from utils.loader import MyPointCloud, ConcatPointClouds
from sklearn.metrics import accuracy_score


def train(model: Net, optim: torch.optim.Optimizer, criterion, dataset: MyPointCloud):
    """Training loop.
    
    At every epoch, we draw a sub-sample from the dataset
    (computing the features on-the-fly).
    """
    model.train()
    
    points, labels, features = dataset.get_sample(size=1600)
    
    optim.zero_grad()
    
    # get feature vector
    x = torch.cat(features, dim=1)
    output = model(x)
    loss = criterion(output, labels)
    loss.backward()
    
    optim.step()
    
    prediction = np.argmax(output.data.numpy(), axis=1)
    accuracy = accuracy_score(labels.data.numpy(), prediction)
    # import ipdb; ipdb.set_trace()
    
    print("Train loss: %.3e. Train accuracy: %.2f" % (loss, 100*accuracy))
    

def validate(model: Net, criterion, val_dataset: MyPointCloud):
    model.eval()
    points, labels, features = val_dataset.get_sample(size=1600)
    with torch.no_grad():
        # get feature vector
        x = torch.cat(features, dim=1)
        output = model(x)
        loss = criterion(output, labels)

        prediction = np.argmax(output.data.numpy(), axis=1)
        accuracy = accuracy_score(labels.data.numpy(), prediction)
        return loss.data.numpy(), accuracy


def save_checkpoint(epoch, model: nn.Module, optimizer, loss, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)


NUM_EPOCHS = 80


if __name__ == "__main__":
    dataset1 = MyPointCloud("data/MiniChallenge/training/MiniLille1.ply")
    dataset2 = MyPointCloud("data/MiniChallenge/training/MiniParis1.ply")
    train_dataset = ConcatPointClouds([dataset1, dataset2])
    val_dataset = MyPointCloud("data/MiniChallenge/training/MiniLille2.ply")
    # import ipdb; ipdb.set_trace()
    model = Net(9)
    optimizer = optim.SGD(model.parameters(), lr=0.04, momentum=.8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = torch.nn.NLLLoss()
    
    VALID_EVERY = 5

    for epoch in tqdm.trange(NUM_EPOCHS):
        print("Epoch %d:", end=' ')
        train(model, optimizer, criterion, train_dataset)
        

        if (epoch+1) % VALID_EVERY == 0:
            val_loss, val_acc = validate(model, criterion, val_dataset)
            print("\tVal loss: %.3e. Val accuracy: %.2f"
                  % (val_loss, 100*val_acc))
            scheduler.step(val_loss)
