import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class Network(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(Network, self).__init__()

        # network
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.conv1 = conv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)

        self.up1 = up(1024, 512)
        self.up2 = up(512, 256)
        self.up3 = up(256, 128)
        self.up4 = up(128, 64)

        self.ending_conv = nn.Conv2d(64, n_classes)

    # define your network here!

    def forward(self, x):
        # implement your forward step
        x0 = self.conv1(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        y0 = self.up1(x4, x3)
        y1 = self.up2(y0, x2)
        y2 = self.up3(y1, x1)
        y3 = self.up4(y2, x0)

        y = self.ending_conv(y3)

        return y


# Reference: https://camo.githubusercontent.com/6b548ee09b97874014d72903c891360beb0989e74b4585249436421558faa89d/68747470733a2f2f692e696d6775722e636f6d2f6a6544567071462e706e67
# https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
class conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class down(nn.module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(nn.MaxPool2d(2), conv(in_channels, out_channels))

    def forward(self, x):
        return self.down(x)


class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up1 = nn.Upsample(2)
        self.up2 = conv(in_channels, out_channels)

    def forward(self, x, skipped_x):
        x = self.up1(x)

        # copy and crop
        skipped_x_crop = TF.center_crop(skipped_x, x.shape[-2:])
        x = torch.cat([x, skipped_x_crop], 1)
        x = self.up2(x)

        return x
