import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
import torchvision
import torchvision.transforms as transforms

import torch.nn.functional as F

from sklearn.cluster import KMeans
import fastcluster

import math

import matplotlib.pyplot as plt
from plotting import plot_phases, plot_results, plot_eval, plot_fourier, plot_phases2, plot_masks, plot_slots, build_color_mask, plot_clusters, plot_clusters2, plot_clusters3, plot_hidden_state_video

from loss_metrics import compute_iou, compute_pixelwise_accuracy, calc_iou, calc_acc
from tqdm import tqdm

from utils import save_weights, set_random_seed


def train(net, dataset_name, trainset, valset, testset, device,
          min_epochs, max_epochs, lr, batch_size, model_type, num_channels_plot, optimizer, weight_decay, num_classes, cp_path, test_seed, patience=5, tolerance=0.001):
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
    net.train()
    if dataset_name == 'pascal-voc':
        loss_func = nn.CrossEntropyLoss(ignore_index=255)
    else:
        loss_func = get_loss()
    loss_foreground = nn.CrossEntropyLoss(ignore_index=0)
    optim = get_optim(net, lr, optimizer, weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0
    epochs = max_epochs
    for epoch in tqdm(range(epochs), desc="Training Epochs"):
        num_train_samples = 0
        epoch_metrics = {'total_loss' : 0,
                         'total_iou' : 0,
                         'total_acc' : 0,
                         'foreground_loss' : 0,
                         'foreground_iou' : 0,
                         'foreground_acc' : 0}
        with tqdm(trainloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False) as batch_loader:
            net.train()
            for x in batch_loader:
                # Train step
                x, x_target = x
                batch_size = x.size(0)
                x = x.to(device) #torch.Size([16, 2, 3, 40, 40]) 
                x_target = x_target.to(device).type(torch.long) #torch.Size([16, 2, 3, 40, 40])
                x_pred, _ = net(x)
                # Loss
                loss = loss_func(x_pred, x_target)

                # Backprop
                net.zero_grad()
                loss.backward()
                optim.step()

                if torch.isnan(loss):
                    print("ERROR: NAN TRAIN LOSS")
                    exit()

                # LOSS
                epoch_metrics['total_loss'] += loss.item() * x.size(0)
                # FOREGROUND LOSS
                loss = loss_foreground(x_pred, x_target)
                epoch_metrics['foreground_loss'] += loss.item() * x.size(0)
                # GET PREDICTED CLASSES
                x_pred = torch.argmax(x_pred, dim=1)
                # IOU
                iou = compute_iou(x_pred, x_target)
                epoch_metrics['total_iou'] += iou
                # ACC
                acc = compute_pixelwise_accuracy(x_pred, x_target)
                epoch_metrics['total_acc'] += acc
                # FOREGROUND IOU
                iou = calc_iou(x_pred, x_target, ignore_class=0)
                epoch_metrics['foreground_iou'] += iou
                # FOREGROUND ACC
                acc = calc_acc(x_pred, x_target, ignore_class=0)
                epoch_metrics['foreground_acc'] += acc

                num_train_samples += x.size(0)
        
        val_metrics = eval_metrics(net, loss_func, loss_foreground, valset, device, epoch, batch_size)

        _ = eval_to_plot(net, loss_func, valset, device, epoch + 1, model_type, batch_size, num_channels_plot, num_classes)

        # Log train metrics
        for metric in epoch_metrics:
            epoch_metrics[metric] = epoch_metrics[metric] / num_train_samples
        #wandb.log({"train_loss": epoch_metrics['total_loss']}, step=epoch + 1)
        #wandb.log({"train_iou": epoch_metrics['total_iou']}, step=epoch + 1)
        #wandb.log({"train_acc": epoch_metrics['total_acc']}, step=epoch + 1)
        #wandb.log({"train_loss_foreground": epoch_metrics['foreground_loss']}, step=epoch + 1)
        #wandb.log({"train_iou_foreground": epoch_metrics['foreground_iou']}, step=epoch + 1)
        #wandb.log({"train_acc_foreground": epoch_metrics['foreground_acc']}, step=epoch + 1)
        wandb.log({
            "train_loss": epoch_metrics['total_loss'],
            "train_iou": epoch_metrics['total_iou'],
            "train_acc": epoch_metrics['total_acc'],
            "train_loss_foreground": epoch_metrics['foreground_loss'],
            "train_iou_foreground": epoch_metrics['foreground_iou'],
            "train_acc_foreground": epoch_metrics['foreground_acc']
        }, step=epoch + 1)

        # Log val metrics
        #wandb.log({"val_loss": val_metrics['total_loss']}, step=epoch + 1)
        #wandb.log({"val_iou": val_metrics['total_iou']}, step=epoch + 1)
        #wandb.log({"val_acc": val_metrics['total_acc']}, step=epoch + 1)
        #wandb.log({"val_loss_foreground": val_metrics['foreground_loss']}, step=epoch + 1)
        #wandb.log({"val_iou_foreground": val_metrics['foreground_iou']}, step=epoch + 1)
        #wandb.log({"val_acc_foreground": val_metrics['foreground_acc']}, step=epoch + 1)
        wandb.log({
            "val_loss": val_metrics['total_loss'],
            "val_iou": val_metrics['total_iou'],
            "val_acc": val_metrics['total_acc'],
            "val_loss_foreground": val_metrics['foreground_loss'],
            "val_iou_foreground": val_metrics['foreground_iou'],
            "val_acc_foreground": val_metrics['foreground_acc']
        }, step=epoch + 1)

        

        # Check if this is the best model so far
        val_loss = val_metrics['total_loss']
        if val_loss < best_val_loss - tolerance:
            best_val_loss = val_loss
            save_weights(net.state_dict(), cp_path)
            patience_counter = 0  # Reset patience counter since we found a new best
            #print("New best model found and saved.")
        else:
            patience_counter += 1
            #print(f"No improvement. Patience counter: {patience_counter}/{patience}")

        # Check for early stopping
        if patience_counter >= patience and epoch + 1 >= min_epochs:
            #print("Early stopping triggered.")
            break

    # Load weights of the best model
    net.load_state_dict(torch.load(cp_path))
    net.eval()

    # Evaluate val set again to do video
    if (not model_type.startswith("unet")) and model_type != 'baseline1_flexible':
        eval_for_video(net, "valset", valset, device, batch_size, num_samples_plot=2, num_channels_plot=num_channels_plot)

    # Evaluate test set
    set_random_seed(test_seed)
    test_metrics = eval_metrics(net, loss_func, loss_foreground, testset, device, epoch, batch_size)
    wandb.log({
        "test_loss": test_metrics['total_loss'],
        "test_iou": test_metrics['total_iou'],
        "test_acc": test_metrics['total_acc'],
        "test_loss_foreground": test_metrics['foreground_loss'],
        "test_iou_foreground": test_metrics['foreground_iou'],
        "test_acc_foreground": test_metrics['foreground_acc']
    }, step=epoch + 1)
    #wandb.log({"test_loss": test_metrics['total_loss']}, step=epoch + 1)
    #wandb.log({"test_iou": test_metrics['total_iou']}, step=epoch + 1)
    #wandb.log({"test_acc": test_metrics['total_acc']}, step=epoch + 1)
    #wandb.log({"test_loss_foreground": test_metrics['foreground_loss']}, step=epoch + 1)
    #wandb.log({"test_iou_foreground": test_metrics['foreground_iou']}, step=epoch + 1)
    #wandb.log({"test_acc_foreground": test_metrics['foreground_acc']}, step=epoch + 1)
    if (not model_type.startswith("unet")) and model_type != 'baseline1_flexible':
        eval_for_video(net, "testset", testset, device, batch_size, num_samples_plot=2, num_channels_plot=num_channels_plot)

    return net

def eval_metrics(net, loss_func, loss_foreground, valset, device, 
                 epoch, batch_size):
    net.eval()
    metrics = {'total_loss' : 0,
               'total_iou' : 0,
               'total_acc' : 0,
               'foreground_loss' : 0,
               'foreground_iou' : 0,
               'foreground_acc' : 0}
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, drop_last=False)
    with torch.no_grad():
        for x in val_loader:
            # Run batch
            x, x_target = x
            x_target = x_target.to(device).type(torch.long)
            x = x.to(device)
            b, c, h, w = x.size()
            x_pred_classifier, _ = net(x)

            # LOSS
            loss = loss_func(x_pred_classifier, x_target)
            metrics['total_loss'] += loss.item() * b
            # FOREGROUND LOSS
            loss = loss_foreground(x_pred_classifier, x_target)
            metrics['foreground_loss'] += loss.item() * b
            # GET PREDICTED CLASSES
            x_pred_classifier = torch.argmax(x_pred_classifier, dim=1)
            # IOU
            iou = compute_iou(x_pred_classifier, x_target)
            metrics['total_iou'] += iou
            # ACC
            acc = compute_pixelwise_accuracy(x_pred_classifier, x_target)
            metrics['total_acc'] += acc
            # FOREGROUND IOU
            iou = calc_iou(x_pred_classifier, x_target, ignore_class=0)
            metrics['foreground_iou'] += iou
            # FOREGROUND ACC
            acc = calc_acc(x_pred_classifier, x_target, ignore_class=0)
            metrics['foreground_acc'] += acc

    num_samples = len(valset)
    for metric in metrics:
        metrics[metric] = metrics[metric] / num_samples
    return metrics

def eval_for_video(net, wandb_name, evalset, device, batch_size, num_samples_plot, num_channels_plot):
    net.eval()
    val_loader = DataLoader(evalset, batch_size=batch_size, shuffle=True, drop_last=True)
    with torch.no_grad():
        for x in val_loader:
            x, x_target = x
            x_target = x_target.to(device).type(torch.long)
            x = x.to(device)
            b, c, h, w = x.size()
            _, y_seq = net(x)

            num_channels_plot = min(num_channels_plot, y_seq.size(2))
            for sample_idx in range(num_samples_plot):
                for curr_c in range(num_channels_plot):
                    plot_hidden_state_video(y_seq, wandb_name, sample_idx=sample_idx, 
                                            channel=curr_c, interval=200, fpath="curr")
            return

def eval_to_plot(net, loss_func, valset, device, epoch, model_type, batch_size, num_channels_plot, num_classes):
    net.eval()
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=True, drop_last=True)
    with torch.no_grad():
        for x in val_loader:
            x, x_target = x
            x_target = x_target.to(device).type(torch.long)
            x = x.to(device)
            b, c, h, w = x.size()
            x_pred_classifier, y_seq = net(x)

            # LOSS
            loss = loss_func(x_pred_classifier, x_target)
            loss = loss.item()

            # CLASSIFIER ARI
            x_pred_classifier = torch.argmax(x_pred_classifier, dim=1)
        
            # SAMPLE RANDOM INDICES TO PLOT
            num_samples = 5
            idx = np.random.choice(range(len(x)), num_samples, replace=False)

            # CLUSTERING
            x = x.detach().cpu().numpy()[idx]
            x = np.transpose(x, (0, 2, 3, 1))
            x_target = x_target.detach().cpu().numpy()[idx]
            x_pred_classifier = x_pred_classifier.detach().cpu().numpy()[idx]
            n_clusters = 21

            # PLOT
            mask_colors = np.random.randint(0, 256, size=(num_classes, 3))
            x_target_mask = build_color_mask(x_target, mask_colors, num_classes)
            x_classifier_mask = build_color_mask(x_pred_classifier, mask_colors, num_classes)
            #plot_clusters3(x, x_target, x_pred_classifier, epoch, num_classes=num_classes)
            plot_clusters3(x, x_target_mask, x_classifier_mask, epoch, num_classes=num_classes)
            
            return y_seq


def get_loss():
    return nn.CrossEntropyLoss()


def get_optim(net, lr, optimizer, weight_decay=0.01):
    if optimizer == 'adam':
        return torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer == 'adamw':
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        print(f"ERROR: {optimizer} is not a valid optimizer.")
        exit()