import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from models.deep_stdfeatures import Net

import tqdm
from utils.loader import MyPointCloud, ConcatPointClouds
from sklearn.metrics import accuracy_score

SAMPLING_SIZE = 4000


def train(model: Net, optim: torch.optim.Optimizer, criterion, dataset):
    """Training loop.
    
    At every epoch, we draw a sub-sample from the dataset
    (computing the features on-the-fly).
    """
    model.train()
    
    points, labels, features = dataset.get_sample(size=SAMPLING_SIZE)
    
    optim.zero_grad()
    
    # get feature vector
    x = torch.cat(features, 1)
    output = model(points, x)
    loss = criterion(output, labels)
    loss.backward()
    
    optim.step()
    
    prediction = np.argmax(output.data.numpy(), axis=1)
    accuracy = accuracy_score(labels.data.numpy(), prediction)
    
    return loss.data.numpy(), accuracy
    

def validate(model: Net, criterion, val_dataset):
    model.eval()
    points, labels, features = val_dataset.get_sample(size=SAMPLING_SIZE)
    with torch.no_grad():
        # get feature vector
        x = torch.cat(features, 1)
        output = model(points, x)
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
VALID_EVERY = 4


if __name__ == "__main__":
    SCALES = [.2, .5]
    dataset1 = MyPointCloud("data/MiniChallenge/training/MiniLille1.ply",
                            multiscale=SCALES)
    dataset2 = MyPointCloud("data/MiniChallenge/training/MiniParis1.ply",
                            multiscale=SCALES)
    print("Train dataset feature scales: %s" % SCALES)
    train_dataset = ConcatPointClouds([dataset1, dataset2])
    print("Train dataset proportions:", train_dataset.proportions)
    val_dataset = MyPointCloud("data/MiniChallenge/training/MiniLille2.ply",
                               multiscale=SCALES)
    num_features = 9 * len(SCALES)
    model = Net(num_features + 3)
    optimizer = optim.SGD(model.parameters(), lr=0.06, momentum=.8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    criterion = torch.nn.NLLLoss()
    
    
    losses_ = []
    accuracies_ = []
    val_losses_ = []
    val_accuracies_ = []

    for epoch in tqdm.tqdm(range(NUM_EPOCHS)):
        print("Epoch %d:" % epoch, end=' ')
        loss, accuracy = train(model, optimizer, criterion, train_dataset)
        
        losses_.append(loss)
        accuracies_.append(accuracy)
        
        print("Train loss: %.3e. Train accuracy: %.2f" % (loss, 100*accuracy))


        if (epoch+1) % VALID_EVERY == 0:
            val_loss, val_acc = validate(model, criterion, val_dataset)
            print("\tVal loss: %.3e. Val accuracy: %.2f"
                  % (val_loss, 100*val_acc))
            scheduler.step(val_loss)

            val_losses_.append(val_loss)
            val_accuracies_.append(val_acc)
    
    import matplotlib.pyplot as plt
    
    plt.style.use("seaborn")
    
    # import ipdb; ipdb.set_trace()
    fig = plt.figure(figsize=(8, 6))
    plt.subplot(211)
    plt.plot(np.arange(NUM_EPOCHS), losses_, lw=1, label="train_loss", marker='+')
    plt.plot(np.linspace(0, NUM_EPOCHS, len(val_losses_)), val_losses_, lw=1, marker='+', label="val_loss")
    plt.legend()
    plt.grid()
    
    plt.subplot(212)
    plt.plot(np.arange(NUM_EPOCHS), accuracies_, lw=1, marker='+', label="train_acc")
    plt.plot(np.linspace(0, NUM_EPOCHS, len(val_losses_)), val_accuracies_, lw=1, marker='+', label="val_acc")
    plt.legend()
    plt.grid()

    plt.show()
