# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    this block is the building block of the residual network. it takes an
    input with in_channels, applies some blocks of convolutional layers
    to reduce it to out_channels and sum it up to the original input,
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        output = F.relu(self.bn1(self.conv1(x)))
        output = self.bn2(self.conv2(output))
        output += self.shortcut(x)
        output = F.relu(output)
        return output

class DenseLayer(nn.Module):

    def __init__(self, in_channels, growth_rate=12, dilation=1):
        super(DenseLayer, self).__init__()

        self.dense_layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, growth_rate * 4, kernel_size=1, stride=1, padding=0),  # In the Huang's
            # DenseNet paper, the bottleneck has a 4*k number of output with k the growth_rate
            nn.BatchNorm2d(growth_rate * 4),
            nn.ReLU(),
            nn.Conv2d(growth_rate * 4, growth_rate, kernel_size=3, stride=1, padding=dilation, dilation=dilation)
        )
    def forward(self, x):
        output = self.dense_layer(x)
        return output

class DenseBlock(nn.Module):

    def __init__(self, in_channels, num_layers=10, growth_rate=12, dilation=1):
        super(DenseBlock, self).__init__()
        self.dense_layers = nn.ModuleList([])

        self.ReLU = nn.ReLU()

        for i in range(num_layers):
            in_layer_channels = in_channels + i * growth_rate
            self.dense_layers.append(DenseLayer(in_layer_channels, growth_rate=growth_rate, dilation=dilation))

    def forward(self, x):
        output = x
        for layer in self.dense_layers:
            dense_output = self.ReLU(layer(output))
            output = torch.cat((output, dense_output), 1)
        return output


class DebugPrint(nn.Module):
    def __init__(self):
        super(DebugPrint, self).__init__()

    def forward(self, x):
        print("Layer shape:", x.shape)
        return x