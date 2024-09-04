
import torch
import torch.nn as nn
import torch.nn.functional as F



class Network(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Network, self).__init__()
        
        # define the layers (7 [conv, relu, pool] layers)
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True) # im/2
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
        )
        
        self.conv6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        
        self.conv7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d()
        )
        
        self.upsample4 = nn.ConvTranspose2d(512, output_channels, 4, stride=2, padding=1, output_padding=0, groups=1, bias=True, dilation=1)
        self.upsample5 = nn.ConvTranspose2d(512, output_channels, 8, stride=4, padding=2, output_padding=0, groups=1, bias=True, dilation=1)
        self.upsample7 = nn.ConvTranspose2d(4096, output_channels, 64, stride=32, padding=16, output_padding=0, groups=1, bias=True, dilation=1)
            
        

    def forward(self, x):

        # implement your forward step
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        conv4 = self.conv4(x)
        conv5 = self.conv5(conv4)
        x = self.conv6(conv5)
        x = self.conv7(x)
        y = self.upsample7(x) + self.upsample5(conv5) + self.upsample4(conv4) 
        
        return y

