"""
Setup the require datasets and a getter
"""

import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import pandas as pd
import numpy as np

class TCGA(Dataset):
  def __init__(self, labels_path, data_path, train_flag):
    labels = pd.read_csv(labels_path)
    data= pd.read_csv(data_path)
    mapping = {"COAD": 0, 'PRAD': 1, 'LUAD': 2, "BRCA": 3, "KIRC": 4}
    labels = labels.replace({'Class': mapping})
    labels = torch.from_numpy(labels.iloc[:,1].values)
    self.training_data = data.iloc[0:600,]
    self.testing_data = data.iloc[600:, ]
    self.training_labels = labels[0:600]
    self.testing_labels = labels[600:]
    self.train_flag = train_flag

  def __len__(self):
    if self.train_flag:
      return self.training_data.shape[0]
    else:
      return self.testing_data.shape[0]

  def __getitem__(self, index):
    if self.train_flag:
      label = self.training_labels[index]
      feature = torch.from_numpy(self.training_data.iloc[index, :].values[1:].astype(np.float32))
    else:
      label = self.testing_labels[index]
      feature = torch.from_numpy(self.testing_data.iloc[index, :].values[1:].astype(np.float32))
    return feature, label

def get_dataloaders(dataset, batch_size):
  if dataset == 'cifar10':
    normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x/255.0 for x in [63.0, 62.1, 66.7]])

    transform_cifar = transforms.Compose([transforms.ToTensor(),normalize,])

    train_dataset = torchvision.datasets.CIFAR10(root = "../../data", train = True, transform = transform_cifar, download = True)
    test_dataset = torchvision.datasets.CIFAR10(root = "../../data", train = False, transform = transform_cifar)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, validation_loader

  elif dataset == 'mnist':
    normalize = transforms.Normalize((0.1307,), (0.3081,))

    transform_mnist = transforms.Compose([transforms.ToTensor(),normalize,])

    train_dataset = torchvision.datasets.MNIST(root = "../../data", train = True, transform = transform_mnist, download = True)
    test_dataset = torchvision.datasets.MNIST(root = "../../data", train = False, transform = transform_mnist)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

    return train_loader, validation_loader

  elif dataset == 'tcga':

    train_dataset = TCGA("data/TCGA/labels.csv", "data/TCGA/data.csv", train_flag = True)
    test_dataset = TCGA("data/TCGA/labels.csv", "data/TCGA/data.csv", train_flag = False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
    validation_loader = torch.utils.data.DataLoader(test_dataset, batch_size = batch_size, shuffle = False)
    return train_loader, validation_loader
  else:
    raise Exception('Unsupported dataset: {0}'.format(dataset))

