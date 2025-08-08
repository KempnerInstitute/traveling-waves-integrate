import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import math
import os
import glob
import pdb

from torch.utils.data import DataLoader

from utils import make_model, set_random_seed
from dataset import load_data

from loss_metrics import compute_iou, compute_pixelwise_accuracy

DATASET_CONFIG = {
    'new_tetronimoes': {
        'x_train_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/new_tetrominoes/train_images.npy", # WRITE ABSOLUTE PATHS
        'y_train_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/new_tetrominoes/train_masks.npy",
        'x_val_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/new_tetrominoes/val_images.npy",
        'y_val_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/new_tetrominoes/val_masks.npy",
        'x_test_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/new_tetrominoes/test_images.npy",
        'y_test_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/new_tetrominoes/test_masks.npy",
        'img_size': 64,
        'channels': 3,
    },
    'multi_mnist': {
        'x_train_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/multi_mnist/train_images.npy",
        'y_train_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/multi_mnist/train_masks.npy",
        'x_val_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/multi_mnist/val_images.npy",
        'y_val_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/multi_mnist/val_masks.npy",
        'x_test_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/multi_mnist/test_images.npy",
        'y_test_path': "/n/ba_lab/Everyone/mjacobs/data/datasets/multi_mnist/test_masks.npy",
        'img_size': 128,
        'channels': 1,
    },
    'mnist': {
        'train_path': '/n/ba_lab/Everyone/mjacobs/data/datasets/mnist/',
        'test_path': '/n/ba_lab/Everyone/mjacobs/data/datasets/mnist/',
        'img_size': 56,  # Image size for resizing
        'channels': 1, # originally 1, but color jitter makes it 3
    },
}

import pdb


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = nn.CrossEntropyLoss()
    
    # Load data
    data_config1 = DATASET_CONFIG['new_tetronimoes']
    data_config2 = DATASET_CONFIG['mnist']
    data_config3 = DATASET_CONFIG['multi_mnist']
    _, _, testset1 = load_data('new_tetronimoes', data_config1)
    _, _, testset2 = load_data('mnist', data_config2)
    _, _, testset3 = load_data('multi_mnist', data_config3)
    testsets = {'new_tetronimoes' : testset1,
                'mnist' : testset2,
                'multi_mnist' : testset3}

    with open("results_scores/background_metrics.txt", "w") as fname:
        for testset in testsets:
            if testset == 'new_tetronimoes':
                num_classes = 6
            else:
                num_classes = 11
            metrics = eval_metrics(loss_func, testsets[testset], device, batch_size=128, num_classes=num_classes)
            print(testset, file=fname)
            print(str(metrics), file=fname)


def eval_metrics(loss_func, valset, device, batch_size, num_classes):
    metrics = {#'total_loss' : 0,
               'total_iou' : 0,
               'total_acc' : 0
              }
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, drop_last=False)
    with torch.no_grad():
        for x in val_loader:
            # Run batch
            x, x_target = x
            x_target = x_target.to(device).type(torch.long)
            x = x.to(device)
            b, c, h, w = x.size()
            x_pred_bg = torch.zeros(b, num_classes, h, w, device=device) # b x C x h x w
            x_pred_bg[:,0] = 1.0

            # LOSS
            #loss = loss_func(x_pred_bg, x_target)
            #metrics['total_loss'] += loss.item() * b
            # GET PREDICTED CLASSES
            x_pred_bg = torch.argmax(x_pred_bg, dim=1)
            # IOU
            iou = compute_iou(x_pred_bg, x_target)#, num_classes)
            metrics['total_iou'] += iou
            # ACC
            acc = compute_pixelwise_accuracy(x_pred_bg, x_target)
            metrics['total_acc'] += acc

    num_samples = len(valset)
    for metric in metrics:
        metrics[metric] = metrics[metric] / num_samples
    return metrics
    

if __name__ == "__main__":
    main()