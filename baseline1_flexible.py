import torch
import torch.nn as nn
import numpy as np
import torch.nn as nn

"""
Convolutional baseline.
"""

class Model(nn.Module):
    def __init__(self, N, c_in, c_mid, c_out, num_classes, min_iters, max_iters, img_size, dt, T, num_layers):
        super(Model, self).__init__()

        self.N = N
        self.c_in = c_in
        self.c_mid = c_mid
        self.c_out = c_out
        self.img_size = img_size
        self.K = int((max_iters - min_iters) / 2) - 1
        self.num_classes = num_classes

        # 3x3, at end its 1x1
        # cornn: 3x3, 3x3, 1x1, 3x3, 

        # Encoder
        assert num_layers >= 1
        layers = [
            nn.Conv2d(c_in, c_mid, kernel_size=3, padding=1, stride=1),
            nn.ReLU()
        ]
        for i in range(num_layers - 1):
            if i == num_layers - 2:
                layers.append(nn.Conv2d(c_mid, c_out, kernel_size=3, padding=1, stride=1))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Conv2d(c_mid, c_mid, kernel_size=3, padding=1, stride=1))
                layers.append(nn.ReLU())
        layers.append(nn.Conv2d(c_out, self.K * c_out, kernel_size=1, padding=0, stride=1))
        self.encoder = nn.Sequential(*layers)

        self.readout = nn.Sequential(
            nn.Linear(self.K * self.c_out, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, self.num_classes)
        )
        
    def forward(self, x):
        x = self.encoder(x) # batch x c_out x N x N
        x = torch.transpose(x, 1, 3)
        x = self.readout(x)
        return torch.transpose(x, 1, 3), None