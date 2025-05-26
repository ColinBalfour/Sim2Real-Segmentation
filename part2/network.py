
import torch
import torch.nn as nn
import torch.nn.functional as F



class Network(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(Network, self).__init__()
        
        # define the layers (7 [conv, relu, pool] layers)
        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(input_channels, 64, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, stride=2, ceil_mode=True) # im/2
        # )
        
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(64, 128, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128, 128, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # )
        
        # self.conv3 = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(256, 256, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # )
        
        # self.conv4 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # )
        
        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(512, 512, 3, padding=1),
        #     nn.ReLU(inplace=True),
        #     nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # )
        
        # self.conv6 = nn.Sequential(
        #     nn.Conv2d(512, 4096, 7),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d()
        # )
        
        # self.conv7 = nn.Sequential(
        #     nn.Conv2d(4096, 4096, 1),
        #     nn.ReLU(inplace=True),
        #     nn.Dropout2d()
        # )
        
        # self.reduce4 = nn.Conv2d(256, output_channels, 1)
        # self.reduce5 = nn.Conv2d(512, output_channels, 1)
        # self.reduce7 = nn.Conv2d(4096, output_channels, 1)
        
        # self.upscore_conv7 = nn.ConvTranspose2d(output_channels, output_channels, 4, stride=2, bias=False)
        # self.upscore_conv5 = nn.ConvTranspose2d(output_channels, output_channels, 4, stride=2, bias=False)
        # self.upscore_conv4 = nn.ConvTranspose2d(output_channels, output_channels, 16, stride=8, bias=False)
        
        # conv1
        self.conv1_1 = nn.Conv2d(input_channels, 64, 3, padding=100)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/2

        # conv2
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/4

        # conv3
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/8

        # conv4
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/16

        # conv5
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)  # 1/32

        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 7)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.score_fr = nn.Conv2d(4096, output_channels, 1)
        self.score_pool3 = nn.Conv2d(256, output_channels, 1)
        self.score_pool4 = nn.Conv2d(512, output_channels, 1)

        self.upscore2 = nn.ConvTranspose2d(
            output_channels, output_channels, 4, stride=2, bias=False)
        self.upscore8 = nn.ConvTranspose2d(
            output_channels, output_channels, 16, stride=8, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(
            output_channels, output_channels, 4, stride=2, bias=False)
            
        

    def forward(self, x):
        # print("SHUFHFFHH")
        # print(x.shape)

        # implement your forward step
        # x = self.conv1(x)
        # x = self.conv2(x)
        # conv4 = self.conv3(x)
        # conv5 = self.conv4(conv4)
        # x = self.conv5(conv5)
        # x = self.conv6(x)
        # x = self.conv7(x)
        # print(conv4.shape, conv5.shape, x.shape)
        
        # upsampled_conv7 = self.upscore_conv7(self.reduce7(x))

        # reduced_conv5 = self.reduce5(conv5)[:, :, 5:5 + upsampled_conv7.size()[2], 5:5 + upsampled_conv7.size()[3]]
        # upsampled_conv5 = self.upscore_conv5(upsampled_conv7 + reduced_conv5)

        # score_conv4 = self.reduce4(conv4)[:, :, 9:9 + upsampled_conv5.size()[2], 9:9 + upsampled_conv5.size()[3]]
        # upscore_conv4 = self.upscore_conv4(upsampled_conv5 + score_conv4)
        
        # y = upscore_conv4[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        # print(y.shape)
        
        h = x
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h = self.pool1(h)

        h = self.relu2_1(self.conv2_1(h))
        h = self.relu2_2(self.conv2_2(h))
        h = self.pool2(h)

        h = self.relu3_1(self.conv3_1(h))
        h = self.relu3_2(self.conv3_2(h))
        h = self.relu3_3(self.conv3_3(h))
        h = self.pool3(h)
        pool3 = h  # 1/8

        h = self.relu4_1(self.conv4_1(h))
        h = self.relu4_2(self.conv4_2(h))
        h = self.relu4_3(self.conv4_3(h))
        h = self.pool4(h)
        pool4 = h  # 1/16

        h = self.relu5_1(self.conv5_1(h))
        h = self.relu5_2(self.conv5_2(h))
        h = self.relu5_3(self.conv5_3(h))
        h = self.pool5(h)

        h = self.relu6(self.fc6(h))
        h = self.drop6(h)

        h = self.relu7(self.fc7(h))
        h = self.drop7(h)

        h = self.score_fr(h)
        h = self.upscore2(h)
        upscore2 = h  # 1/16

        h = self.score_pool4(pool4)
        h = h[:, :, 5:5 + upscore2.size()[2], 5:5 + upscore2.size()[3]]
        score_pool4c = h  # 1/16

        h = upscore2 + score_pool4c  # 1/16
        h = self.upscore_pool4(h)
        upscore_pool4 = h  # 1/8

        h = self.score_pool3(pool3)
        h = h[:, :,
              9:9 + upscore_pool4.size()[2],
              9:9 + upscore_pool4.size()[3]]
        score_pool3c = h  # 1/8

        h = upscore_pool4 + score_pool3c  # 1/8

        h = self.upscore8(h)
        y = h[:, :, 31:31 + x.size()[2], 31:31 + x.size()[3]].contiguous()
        # print(y.shape)

        return y