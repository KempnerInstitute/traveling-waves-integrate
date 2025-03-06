""" Full assembly of the parts to form the complete network """

# Taken from repo online
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py

import torch
import torch.nn as nn


from unet_parts import DoubleConv, Down, Up, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, c_mid):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.c_mid = c_mid

        bilinear = False
        self.inc = (DoubleConv(n_channels, c_mid))
        self.down1 = (Down(c_mid * 1, c_mid * 2))
        self.down2 = (Down(c_mid * 2, c_mid * 4))
        self.down3 = (Down(c_mid * 4, c_mid * 8))
        self.down4 = (Down(c_mid * 8, c_mid * 16))

        self.up1 = (Up(c_mid * 16, c_mid * 8, bilinear))
        self.up2 = (Up(c_mid * 8, c_mid * 4, bilinear))
        self.up3 = (Up(c_mid * 4, c_mid * 2, bilinear))
        self.up4 = (Up(c_mid * 2, c_mid * 1, bilinear))
        self.outc = (OutConv(c_mid, n_classes))

    def forward(self, x):
        x1 = self.inc(x) # 128 -> 128
        x2 = self.down1(x1) # 128 -> 64
        x3 = self.down2(x2) # 64 -> 32
        x4 = self.down3(x3) # 32 -> 16
        x5 = self.down4(x4) # 16 -> 8
        x = self.up1(x5, x4) # 8 -> 16
        x = self.up2(x, x3) # 16 -> 32
        x = self.up3(x, x2) # 32 -> 64
        x = self.up4(x, x1) # 64 -> 128
        logits = self.outc(x) # 128 -> 128
        return logits, None