import torch
import torch.nn as nn
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
from functools import partial

# convoluation block consist 2 3*3 convolution layer with relu activation function and batch normalization, the first convoluaiton layer has a stide of 1, the 
# second one has a stride of 2

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=True):
        super().__init__()
        stride = 2 if downsample else 1

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut projection if dimensions differ or downsampling is needed
        if downsample or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out



class M3ENet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_class = num_classes
        self.block1 = nn.Sequential(
            ResBlock(in_channels=1, out_channels=8, downsample=True),
            ResBlock(in_channels=8, out_channels=32, downsample=True),
            ResBlock(in_channels=32, out_channels=64, downsample=True),
        )
        self.fc1 = nn.Linear(2304, 256)
        self.block2 = nn.Sequential(
            ResBlock(in_channels=1, out_channels=8, downsample=True),
            ResBlock(in_channels=8, out_channels=32, downsample=True),
            ResBlock(in_channels=32, out_channels=64, downsample=True),
        )
        self.fc2 = nn.Linear(2304, 256)
        self.block3 = nn.Sequential(
            ResBlock(in_channels=1, out_channels=8, downsample=True),
            ResBlock(in_channels=8, out_channels=32, downsample=True),
            ResBlock(in_channels=32, out_channels=64, downsample=True),
        )
        self.fc3 = nn.Linear(2304, 256)
        self.block_rgb = nn.Sequential(
            ResBlock(in_channels=1, out_channels=8, downsample=True),
            ResBlock(in_channels=8, out_channels=32, downsample=True),
            ResBlock(in_channels=32, out_channels=32, downsample=True),
        )
        # self.fc_rgb = nn.Linear(2304, 256)

        self.block_rgb_c1 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=64, downsample=False),
            ResBlock(in_channels=64, out_channels=64, downsample=False),
            ResBlock(in_channels=64, out_channels=64, downsample=False),
        )
        self.fc_rgb_c1 = nn.Linear(2304, 256)

        self.block_rgb_c2 = nn.Sequential(
            ResBlock(in_channels=64, out_channels=64, downsample=False),
            ResBlock(in_channels=64, out_channels=64, downsample=False),
            ResBlock(in_channels=64, out_channels=64, downsample=False),
        )
        self.fc_rgb_c2 = nn.Linear(2304, 256)


        self.Block1 = nn.Sequential(nn.Linear(256*5, 1024), nn.ReLU(inplace=True))
        self.Block2 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(inplace=True))
        self.Block3 = nn.Sequential(nn.Linear(1024, num_classes))


    def forward(self, x1, x2, x3, rgb1, rgb2, rgb3):
        x1 = self.block1(x1)   # x1: [batch_size, 64, 6, 6]
        x1 = x1.view(-1, 2304)
        x1 = self.fc1(x1)

        x2 = self.block2(x2)   # x2: [batch_size, 64, 6, 6]
        x2 = x2.view(-1, 2304)
        x2 = self.fc2(x2)

        x3 = self.block3(x3)   # x1: [batch_size, 64, 6, 6]
        x3 = x3.view(-1, 2304)
        x3 = self.fc3(x3)

        rgb1 = self.block_rgb(rgb1)   # x1: [batch_size, 64, 6, 6]
        rgb2 = self.block_rgb(rgb2)   # x1: [batch_size, 64, 6, 6]
        rgb3 = self.block_rgb(rgb3)   # x1: [batch_size, 64, 6, 6]

        rgb_c1 = torch.cat([rgb1, rgb2], dim=1)
        rgb_c1 = self.block_rgb_c1(rgb_c1)
        rgb_c1 = rgb_c1.view(-1, 2304)
        rgb_c1 = self.fc_rgb_c1(rgb_c1)

        rgb_c2 = torch.cat([rgb3, rgb2], dim=1)
        rgb_c2 = self.block_rgb_c2(rgb_c2)
        rgb_c2 = rgb_c2.view(-1, 2304)
        rgb_c2 = self.fc_rgb_c2(rgb_c2)


        out = torch.cat([x1, x2, x3, rgb_c1, rgb_c2], dim=1)


        out = self.Block1(out)
        out = self.Block2(out)
        out = self.Block3(out)

        return out





