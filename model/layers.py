import torch
import torch.nn as nn
import torch.nn.functional as F


class UnetConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, batch_norm=True):
        super().__init__()

        if batch_norm:
            self.DoubleConv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
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


class UnetDown(nn.Module):
    # maxpooling then double conv
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.Down = nn.Sequential(
            nn.MaxPool2d(2),
            UnetConv2d(in_channels, out_channels)
        )

    def forward(self, x):
        return self.Down(x)


class UnetUp(nn.Module):
    # upscaling then double conv
    def __init__(self, in_channels, out_channels, up_sample=True):
        super().__init__()

        if up_sample:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.Conv2d = UnetConv2d(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diff_y = torch.tensor([x2.size()[2] - x1.size()[2]])
        diff_x = torch.tensor([x2.size()[3] - x1.size()[3]])
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        out = torch.cat([x2, x1], dim=1)
        return self.Conv2d(out)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.out = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.out(x)
