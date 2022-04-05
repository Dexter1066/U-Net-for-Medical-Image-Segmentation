from model.layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up_sample = up_sample

        self.inc = UnetConv2d(in_channels, 64)
        self.down1 = UnetDown(64, 128)
        self.down2 = UnetDown(128, 256)
        self.down3 = UnetDown(256, 512)
        self.down4 = UnetDown(512, 512)

        self.up1 = UnetUp(1024, 256, up_sample)
        self.up2 = UnetUp(512, 128, up_sample)
        self.up3 = UnetUp(256, 64, up_sample)
        self.up4 = UnetUp(128, 64, up_sample)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        output = self.outc(x)
        return output
