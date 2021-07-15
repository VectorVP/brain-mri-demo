import os
import torch
import nilearn

import torch.nn as nn
import torch.utils.data as torch_data
import torch.nn.functional as F

from nilearn import plotting
from sklearn.model_selection import train_test_split, StratifiedKFold

from models.3dcnn import MriNet, MriData



def visualize_data(data_dir, nii_file_path):
    img = nilearn.image.load_img(os.path.join(data_dir, nii_file_path))
    img_array = nilearn.image.get_data(img)
    file_name = nii_file_path[:-4]
    plotting.plot_anat(img, output_file=f'{file_name}.png')
    print(f'Shape of {nii_file_path}: ', img_array.shape)


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

