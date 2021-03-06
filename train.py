import os
import torch
import nilearn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch.nn as nn
import torch.utils.data as torch_data
import torch.nn.functional as F

from torchsummary import summary
from sklearn.model_selection import train_test_split, StratifiedKFold
from tqdm import tqdm

from models.3dcnn import MriNet, MriData


def get_accuracy(net, data_loader, device):
    net.eval()
    correct = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        out = net(data)
        pred = out.data.max(1)[1] # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        del data, target
    accuracy = 100. * correct / len(data_loader.dataset)
    return accuracy.item()


def get_loss(net, data_loader, device):
    net.eval()
    loss = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        out = net(data)
        loss += criterion(out, target).item()*len(data)

        del data, target, out

    return loss / len(data_loader.dataset)


def train(epochs, net, device, data_dir, criterion, optimizer, train_loader, val_loader, scheduler=None, verbose=True, save=False):
    CHECKPOINTS_DIR =  'checkpoints'
    best_val_loss = 100000
    best_model = None
    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    train_loss_list.append(get_loss(net, train_loader))
    val_loss_list.append(get_loss(net, val_loader))
    train_acc_list.append(get_accuracy(net, train_loader))
    val_acc_list.append(get_accuracy(net, val_loader))
    if verbose:
        print('Epoch {:02d}/{} || Loss:  Train {:.4f} | Validation {:.4f}'.format(0, epochs, train_loss_list[-1], val_loss_list[-1]))

    net.to(device)
    for epoch in tqdm(range(1, epochs+1)):
        net.train()
        for X, y in train_loader:
            # Perform one step of minibatch stochastic gradient descent
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            out = net(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            del X, y, out, loss #freeing gpu space

        # define NN evaluation, i.e. turn off dropouts, batchnorms, etc.
        net.eval()
        for X, y in val_loader:
            # Compute the validation loss
            X, y = X.to(device), y.to(device)
            out = net(X)
            del X, y, out #freeing gpu space

        if scheduler is not None:
            scheduler.step()

        train_loss_list.append(get_loss(net, train_loader))
        val_loss_list.append(get_loss(net, val_loader))
        train_acc_list.append(get_accuracy(net, train_loader))
        val_acc_list.append(get_accuracy(net, val_loader))

        if save and val_loss_list[-1] < best_val_loss:
            torch.save(net.state_dict(), os.path.join(CHECKPOINTS_DIR, 'best_model'))
        freq = 1
        if verbose and epoch%freq==0:
            print('Epoch {:02d}/{} || Loss:  Train {:.4f} | Validation {:.4f}'.format(epoch, epochs, train_loss_list[-1], val_loss_list[-1]))

    return train_loss_list, val_loss_list, train_acc_list, val_acc_list


def main():
    # TODO: make it elegant and convenient
    EPOCHS = 300
    data_dir = 'anat'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    c = 32

    model = MriNet(c).to(device)
    X, y = np.load(os.path.join(data_dir, 'tensors.npy')), np.load(os.path.join(data_dir, 'labels.npy'))
    X = X[:, np.newaxis, :, :, :]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    train_dataset = MriData(X_train, y_train)
    test_dataset = MriData(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=45, shuffle=True)  #45 - recommended value for batchsize
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=28, shuffle=False)

    criterion = nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1)

    train(EPOCHS, model, device, data_dir, criterion, optimizer, train_loader, val_loader, scheduler=scheduler, verbose=False, save=True)


if __name__ == "__main__":
    main()
