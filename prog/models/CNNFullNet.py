
import torch.nn as nn
from models.CNNBaseModel import CNNBaseModel
from models.CNNBlocks import DenseBlock
import numpy as np


class FullNet(CNNBaseModel):
    """
    """

    def __init__(self, num_classes=4, in_channels=1, num_layers=6, growth_rate=12, init_weights=True):
        """
        Builds vanilla CNN for segmentation.
        Args:
            num_classes(int): number of classes. default 10(cifar10 or svhn)
            in_channels(int): number of channels (ex. modalities) as input
            init_weights(bool): when true uses _initialize_weights function to
            num_layers(int): number of layer in the dense block
            growth_rate(int): number of channels added in the concatenation per layer
            initialize network's weights.
        """
        super(FullNet, self).__init__(num_classes, init_weights)
        # encoder

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.out_channels = num_classes

        dilations = np.array([1, 2, 4, 8, 16, 4, 1]) #FullNet paper dilatation growth factors

        self.blocks = nn.Sequential()

        in_dense_channels = 24 #FullNet paper set it to 24
        self.first_conv = nn.Conv2d(self.in_channels, in_dense_channels, kernel_size=3, padding=1)

        i = 1
        for dilation in dilations:
            self.blocks.add_module('dense_block_%d' % i, DenseBlock(in_channels=in_dense_channels, num_layers=num_layers, growth_rate=growth_rate, dilation=dilation))
            temp_in_channels = in_dense_channels + num_layers * growth_rate
            self.blocks.add_module('conv_1x1_%d' % i, nn.Conv2d(temp_in_channels, temp_in_channels, kernel_size=1, padding=0))
            in_dense_channels = temp_in_channels
            i = i+1

        self.last_conv = nn.Conv2d(in_dense_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        """
        Forward pass of the model
        Args:
            x: Tensor
        """
        out = self.first_conv(x)
        out = self.blocks(out)
        out = self.last_conv(out)
        return out
