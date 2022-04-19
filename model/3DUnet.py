from layers import *
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet3d(nn.Module):
    def __init__(self, in_channels, num_filters, out_channels, up_sample=True, dropout=False):
        super(UNet3d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters = num_filters
        self.up_sample = up_sample
        self.dropout = dropout

        self.down1 = UnetDown3D(in_channels, num_filters)
        self.down2 = UnetDown3D(num_filters, num_filters * 2)
        self.down3 = UnetDown3D(num_filters * 2, num_filters * 4)
        self.down4 = UnetDown3D(num_filters * 4, num_filters * 8)

        self.bottleneck = UnetConv3D(num_filters * 8, num_filters * 16)

        self.up1 = UnetUp3D(num_filters * 16, num_filters * 8, up_sample)
        self.up2 = UnetUp3D(num_filters * 8, num_filters * 4, up_sample)
        self.up3 = UnetUp3D(num_filters * 4, num_filters * 2, up_sample)
        self.up4 = UnetUp3D(num_filters * 2, num_filters, up_sample)

        self.outc = UnetConv3D(num_filters, out_channels)

        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, x):
        conv1, x = self.down1(x)
        conv2, x = self.down2(x)
        conv3, x = self.down3(x)
        conv4, x = self.down4(x)
        x = self.bottleneck(x)

        if self.dropout:
            x = self.dropout(x)

        x = self.up1(x, conv4)
        x = self.up2(x, conv3)
        x = self.up3(x, conv2)
        x = self.up4(x, conv1)

        if self.dropout:
            x = self.dropout(x)

        output = self.outc(x)

        return output
