import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
import torch.nn.functional as F
from itertools import combinations
from torchvision import datasets, transforms
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from torchvision import datasets

from torch.utils.data import random_split


class MNISTSegmentationDataset(Dataset):

    def __init__(self, mnist_dataset, image_size):
        super().__init__()
        # Load MNIST dataset
        self.image_size = image_size
        #self.mnist = datasets.MNIST(data_path, train=train, download=False,
        #                          transform=transforms.ToTensor())
        self.mnist = mnist_dataset
        
    def __len__(self):
        return len(self.mnist)
    
    def __getitem__(self, idx):
        img, label = self.mnist[idx]
        # Upsample image by 2x using bilinear interpolation
        img = F.interpolate(img.unsqueeze(0), size=(self.image_size, self.image_size), 
                          mode='bilinear', align_corners=False).squeeze(0)
        
        # Convert image to binary mask where 1=digit, 0=background
        mask = (img[0] > 0.5).long()  # Shape: (56, 56)
        
        # Set digit pixels to label+1 (so background=0, digit1=1, digit2=2, etc)
        mask[mask == 1] = label + 1
        
        return img, mask


class ShapeDataset(Dataset):
    def __init__(self, x, labels):
        self.x = x
        self.labels = labels

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.labels[idx]


#def load_data(dataset, data_config, num_train, num_test, scale_min=0.7, transform_set='set1', normalize=True):
def load_data(dataset, data_config):
    if dataset == 'new_tetronimoes':
        x_train, y_train = load_new_tetrominoes(data_config['x_train_path'], 
                                                data_config['y_train_path'])
        x_val, y_val = load_new_tetrominoes(data_config['x_val_path'], 
                                              data_config['y_val_path'])
        x_test, y_test = load_new_tetrominoes(data_config['x_test_path'], 
                                              data_config['y_test_path'])
        trainset = ShapeDataset(x_train, y_train)
        valset = ShapeDataset(x_val, y_val)
        testset = ShapeDataset(x_test, y_test)
        return trainset, valset, testset
    elif dataset == 'mnist':
        torch.manual_seed(42)
        trainset = datasets.MNIST(data_config['train_path'], train=True, download=False,
                                  transform=transforms.ToTensor())
        testset = datasets.MNIST(data_config['test_path'], train=False, download=False,
                                  transform=transforms.ToTensor())
        total_size = len(trainset)          # Total number of samples in the dataset
        train_size = int(0.85 * total_size)  # 85% for training
        val_size = total_size - train_size
        trainset, valset = random_split(trainset, [train_size, val_size])
        trainset = MNISTSegmentationDataset(trainset, data_config['img_size'])
        valset = MNISTSegmentationDataset(valset, data_config['img_size'])
        testset = MNISTSegmentationDataset(testset, data_config['img_size'])
        return trainset, valset, testset
    elif dataset == 'multi_mnist':
        x_train, y_train = load_multi_mnist(data_config['x_train_path'], 
                                            data_config['y_train_path'])
        x_val, y_val = load_multi_mnist(data_config['x_val_path'], 
                                        data_config['y_val_path'])
        x_test, y_test = load_multi_mnist(data_config['x_test_path'], 
                                          data_config['y_test_path'])
        trainset = ShapeDataset(x_train, y_train)
        valset = ShapeDataset(x_val, y_val)
        testset = ShapeDataset(x_test, y_test)
        return trainset, valset, testset
    else:
        print(f"ERROR: {dataset} is not a valid dataset.")
        exit()
    return x_train, y_train, x_test, y_test


def load_multi_mnist(x_path, y_path):
    x, y = np.load(x_path), np.load(y_path)
    x = np.expand_dims(x, axis=1) / 255.0
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    return x.type(torch.float), y


def load_new_tetrominoes(x_path, y_path):
    x, y = np.load(x_path), np.load(y_path)
    x = np.transpose(x, (0, 3, 1, 2)) / 255.0
    x, y = torch.from_numpy(x), torch.from_numpy(y)
    return x.type(torch.float), y