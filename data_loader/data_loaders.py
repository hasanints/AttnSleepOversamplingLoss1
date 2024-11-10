import torch
from torch.utils.data import Dataset
import os
import numpy as np
from imblearn.over_sampling import SMOTE

class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset):
        super(LoadDataset_from_numpy, self).__init__()

        # load files
        X_train = np.load(np_dataset[0])["x"]
        y_train = np.load(np_dataset[0])["y"]

        for np_file in np_dataset[1:]:
            X_train = np.vstack((X_train, np.load(np_file)["x"]))
            y_train = np.append(y_train, np.load(np_file)["y"])

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def data_generator_np(training_files, subject_files, batch_size):
    # Load the datasets
    train_dataset = LoadDataset_from_numpy(training_files)
    test_dataset = LoadDataset_from_numpy(subject_files)

    # Reshape X_train to 2D before applying SMOTE
    X_train, y_train = train_dataset.x_data.numpy(), train_dataset.y_data.numpy()
    n_samples, n_channels, seq_len = X_train.shape
    X_train_2d = X_train.reshape(n_samples, n_channels * seq_len)

    # Apply SMOTE to the 2D reshaped data
    smote = SMOTE(random_state=42)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train_2d, y_train)

    # Reshape back to 3D after SMOTE
    X_train_oversampled = X_train_oversampled.reshape(-1, n_channels, seq_len)

    # Convert back to PyTorch tensors
    train_dataset.x_data = torch.from_numpy(X_train_oversampled).float()
    train_dataset.y_data = torch.from_numpy(y_train_oversampled).long()

    # Calculate class ratios for further class balancing if needed
    all_ys = np.concatenate((y_train_oversampled, test_dataset.y_data.numpy()))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0)

    return train_loader, test_loader, counts
