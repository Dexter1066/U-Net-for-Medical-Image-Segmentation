import torch
import torch.nn as nn


class UnetConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
        super(UnetConv2d).__init__()

        if batch_norm:
            self.DoubleConv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.DoubleConv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.ReLU(inplace=True)
            )
    
    def forward(self, x):
        return self.DoubleConv(x)


class UnetUp(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode, padding=1, batch_norm=True):
        super(UnetUp).__init__()

        if up_mode == 'upconv':
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        elif up_mode == 'upsample':
            self.up = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        self.Conv2d = UnetConv2d(in_channels, out_channels, padding, batch_norm)

    def CenterCrop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_x = (layer_width - target_size[1]) // 2
        diff_y = (layer_height - target_size[0]) // 2
        return layer[
            :, :, diff_y : (diff_y + target_size[0]), diff_x : (diff_x + target_size[1])
        ]

    def forward(self, x, bridge):
        up = self.up(x)
        crop = self.CenterCrop(bridge, up.shape[2:])
        out = torch.cat([up, crop], 1)
        return self.Conv2d(out)
