import torch

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import nn

import torchvision
from torchvision import datasets
from torchvision.transforms import v2

from torch.utils.data import Dataset, DataLoader
from PIL import Image

#hyperparameters


#model adjusted for small size of cifar10 images (32x32)
class Block(nn.Module): 
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample = None):
        super().__init__()

        

        self.relu = nn.ReLU()
        self.downsample = downsample


        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x    
    
        out = self.conv1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)

        return out


class ResNet18(nn.Module):
    
    def __init__(self, block, layers: list, num_classes:int = 10):
        super().__init__()

        self.in_channels = 64

        #start the model with a modified layer for our small images
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        #layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, out_channels: int, blocks, stride: int):
        downsample = None

        if stride != 1 or self.in_channels != out_channels*block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels*block.expansion, kernel_size = 1, stride = stride),
                nn.BatchNorm2d(out_channels*block.expansion)
            )
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)

        return x
