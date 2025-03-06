import torch
import torch.nn as nn


def fft_readout(net, y_seq, B, H, W):
    fft_vals = torch.fft.rfft(y_seq, dim=1) # (B, K, c_out, H, W)
    fft_mag = torch.abs(fft_vals) # (B, K, c_out, H, W)
    fft_mag = fft_mag.reshape(B, -1, H, W) # (B, K*c_out, H, W)
    logits = net.fc_readout(torch.transpose(fft_mag, 1, 3)) # (B, W, H, n_classes)
    logits = torch.transpose(logits, 1, 3) # (B, n_classes, H, W)
    return logits

def linear_readout(net, y_seq, B, H, W):
    y_seq = y_seq.reshape(B, net.T, net.c_out, -1)
    y_seq = y_seq.transpose(1, 3)
    fft_vals = net.fc_time(y_seq)
    #fft_mag = torch.abs(fft_vals)
    fft_mag = fft_vals.transpose(1, 3) # (B, K, C, H*W)
    fft_mag = fft_mag.reshape(B, net.K * net.c_out, -1)
    fft_mag = fft_mag.transpose(1, 2)
    logits = net.fc_readout(fft_mag)
    logits = logits.transpose(1, 2)
    logits = logits.view(B, net.n_classes, net.N, net.N)
    return logits

def last_readout(net, y_seq, B, H, W):
    hy = y_seq[:,-1] # B T C H W -> B C H W
    logits = net.fc_readout(torch.transpose(hy, 1, 3))
    logits = torch.transpose(logits, 1, 3) # (B, n_channels, H, W)
    return logits

def mean_time_readout(net, y_seq, B, H, W):
    # y_seq: B T C H W
    hy = y_seq.mean(1) # B C H W
    logits = net.fc_readout(torch.transpose(hy, 1, 3))
    logits = torch.transpose(logits, 1, 3) # (B, n_channels, H, W)
    return logits

def max_time_readout(net, y_seq, B, H, W):
    # y_seq: B T C H W
    hy, _ = y_seq.max(1) # B C H W
    logits = net.fc_readout(torch.transpose(hy, 1, 3))
    logits = torch.transpose(logits, 1, 3) # (B, n_channels, H, W)
    return logits