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


class MriData(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(MriData, self).__init__()
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y).long()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


hidden = lambda c_in, c_out: nn.Sequential(
    nn.Conv3d(c_in, c_out, (3,3,3)), # Convolutional layer
    nn.BatchNorm3d(c_out), # Batch Normalization layer
    nn.ReLU(), # Activational layer
    nn.MaxPool3d(2) # Pooling layer
)


class MriNet(nn.Module):
    def __init__(self, c):
        super(MriNet, self).__init__()
        self.hidden1 = hidden(1, c)
        self.hidden2 = hidden(c, 2*c)
        self.hidden3 = hidden(2*c, 4*c)
        self.linear = nn.Linear(128*5*7*5, 2)
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=1)
        return x


def get_accuracy(net, data_loader):
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


def get_loss(net, data_loader):
    net.eval()
    loss = 0
    for data, target in data_loader:
        data = data.to(device)
        target = target.to(device)

        out = net(data)
        loss += criterion(out, target).item()*len(data)

        del data, target, out

    return loss / len(data_loader.dataset)


def train(epochs, net, data_dir, criterion, optimizer, train_loader, val_loader, scheduler=None, verbose=True, save=False):
    CHECKPOINTS_DIR =  os.path.join(data_dir, 'checkpoints')
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


def k_fold_validation():
    EPOCHS = 200
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cross_vall_acc_list = []
    j = 0

    for train_index, test_index in skf.split(X, y):
        print('Doing {} split'.format(j))
        j += 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        train_dataset = MriData(X_train, y_train)
        test_dataset = MriData(X_test, y_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=45, shuffle=True)  #45 - recommended value for batchsize
        val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=28, shuffle=False)

        torch.manual_seed(1)
        np.random.seed(1)

        c = 32
        model = MriNet(c).to(device)
        criterion = nn.NLLLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1)

        train(EPOCHS, model, device, criterion, optimizer, train_loader, val_loader, scheduler=scheduler, save=False, verbose=False)
        cross_vall_acc_list.append(get_accuracy(model, val_loader))

    print('Average cross-validation accuracy (3-folds):', sum(cross_vall_acc_list)/len(cross_vall_acc_list))


def main():
    # TODO: make it elegant and convenient
    data_dir = 'anat'
    EPOCHS = 200

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(1)
    np.random.seed(1)

    c = 32
    model = MriNet(c).to(device)
    summary(model, (1, 58, 70, 58))

    X, y = np.load(os.path.join(data_dir, 'tensors.npy')), np.load(os.path.join(data_dir, 'labels.npy'))
    X = X[:, np.newaxis, :, :, :]
    print(X.shape, y.shape)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    train_dataset = MriData(X_train, y_train)
    test_dataset = MriData(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=45, shuffle=True)  #45 - recommended value for batchsize
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=28, shuffle=False)

    dataset = MriData(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=45, shuffle=True)  #45 - recommended value for batchsize

    torch.manual_seed(1)
    np.random.seed(1)

    model = MriNet(c).to(device)
    criterion = nn.NLLLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15], gamma=0.1)

    train(EPOCHS, model, data_dir, criterion, optimizer, loader, loader, scheduler=scheduler, save=True, verbose=False)


if __name__ == "__main__":
    main()
