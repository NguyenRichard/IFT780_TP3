# -*- coding:utf-8 -*-

"""
University of Sherbrooke
Date:
Authors: Mamadou Mountagha BAH & Pierre-Marc Jodoin
License:
Other: Suggestions are welcome
"""

import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel


class CNNet(CNNBaseModel):
    """
    """

    def __init__(self, num_classes=4, in_channels=1, init_weights=True):
        """
        Builds vanilla CNN for segmentation.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            in_channels(int): number of channels (ex. modalities) as input
            init_weights(bool): when true uses _initialize_weights function to
            initialize network's weights.
        """
        super(CNNet, self).__init__()
        # encoder

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = num_classes

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=32,
                      kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=128,
                      kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=256,
                      kernel_size=3, padding="same"),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=self.out_channels,
                      kernel_size=3, padding="same"),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(self.out_channels),
        )

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        x = self.conv_layers(x)
        return x
