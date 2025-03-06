import torch
import numpy as np
import math
import imageio
import random
import torch.nn as nn


def set_random_seed(seed):
    """function sets the seed value
    Args:
        seed (int): seed value
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_fc_readout(readout_type, K, c_out, n_classes, T, c_mid):
    if readout_type == 'linear_smaller':
        return nn.Sequential(
            nn.Linear(K * c_out, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes)
        )
    elif readout_type == 'linear_smaller2':
        return nn.Sequential(
            nn.Linear(K * c_out, n_classes)
        )
    elif readout_type == 'linear_smaller3':
        return nn.Sequential(
            nn.Linear(K * c_out, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    elif readout_type == 'linear_smaller4':
        return nn.Sequential(
            nn.Linear(K * c_out, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, n_classes)
        )
    elif readout_type != 'last' and readout_type != 'mean_time' and readout_type != 'max_time':
        return nn.Sequential(
            nn.Linear(K * c_out, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
    else:
        return nn.Sequential(
            nn.Linear(c_out, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )

def calc_params(model):
    # Total number of parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Number of trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params, total_params

def make_model(device, model_type, num_classes,
               N, dt1, min_iters, max_iters, c_in, c_mid, c_out, 
               rnn_kernel, img_size, kernel_init, cell_type, num_layers,
               readout_type):

    # CHOOSE READOUT
    from readouts import fft_readout, linear_readout, last_readout, mean_time_readout, max_time_readout
    if readout_type == 'last':
        readout = last_readout
    elif readout_type == 'fft':
        readout = fft_readout
    elif readout_type == 'linear' or readout_type == 'linear_smaller' or readout_type == 'linear_smaller2' or readout_type == 'linear_smaller3' or readout_type == 'linear_smaller4':
        readout = linear_readout
    elif readout_type == 'mean_time':
        readout = mean_time_readout
    elif readout_type == 'max_time':
        readout = max_time_readout
    else:
        raise ValueError(f"Invalid readout {readout}. Expected last, fft, linear, or stft")
    fc_readout = make_fc_readout(readout_type, max_iters//2 + 1, c_out, num_classes, max_iters - min_iters, c_mid)

    if model_type == 'cornn_model2':
        from cornn_model2 import Model
        net = Model(N, c_in, c_mid, c_out, num_classes, min_iters, max_iters, img_size, dt1, max_iters, readout, fc_readout)
    elif model_type == 'conv_recurrent2':
        from conv_recurrent2 import Model
        net = Model(N, c_in, c_mid, c_out, num_classes, min_iters, max_iters, img_size, dt1, max_iters, cell_type, readout, fc_readout)
    elif model_type == 'baseline1_flexible':
        from baseline1_flexible import Model
        net = Model(N, c_in, c_mid, c_out, num_classes, min_iters, max_iters, img_size, dt1, max_iters, num_layers)
    elif model_type == 'unet2':
        from unet2 import UNet as Model
        net = Model(c_in, num_classes, c_mid)
    net = net.to(device)
    return net


def save_weights(weights, cp_path):
    torch.save(weights, cp_path)


def save_model(net, cp_path):
    torch.save(net.state_dict(), cp_path)


def load_model(net, cp_path):
    net.load_state_dict(torch.load(cp_path), strict=False)
    net.eval()
    return net