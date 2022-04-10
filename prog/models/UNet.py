#Source : https://github.com/milesial/Pytorch-UNet

import torch.nn as nn

from models.CNNFullNet import FullNet
from models.UnetParts import *
from os.path import join

class UNet(nn.Module):
    def __init__(self, num_classes=4, num_channels=1, bilinear=False,dense = False):
        super(UNet, self).__init__()
        self.n_channels = num_channels
        self.n_classes = num_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(num_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        
        if dense : #Adding a dense layer to the U-Net
          self.dense = FullNet(1024, 1024, 4)
        
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, num_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def save(self, path="Unet", filename=None):
        """
        Save the model into the filename
        :arg
            filename: file in which to save the model
        """
        filename = filename if filename is not None \
            else self.__class__.__name__ + '.pt'
        torch.save(self.state_dict(), join(path, filename))

    def load_weights(self, file_path="Unet"):
        """
        Load the model's weights saved into the filename
        :arg
            file_path: path file where model's weights are saved
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(file_path, map_location=device))
