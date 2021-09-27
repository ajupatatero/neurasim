import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.autograd import Variable
from collections import OrderedDict
import random

#Create the model

class _ConvBlockInit(nn.Module):
    """
    First block - quarter scale.
    Four Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    Optional dropout before final Conv2d layer
    ReLU after first two Conv2d layers, not after last two - predictions can be +ve or -ve
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlockInit, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 0),
            nn.ReLU(),
        ]

        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlockMid(nn.Module):
    """
    Second block - half scale.
    Six Conv2d layers. First one kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv2d layer
    ReLU after first four Conv2d layers, not after last two - predictions can be +ve or -ve
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlockMid, self).__init__()
        layers = [
            nn.MaxPool2d(2),
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 0),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlockUp(nn.Module):
    """
    Third block - full scale.
    Six Conv2d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv2d layer
    ReLU after first four Conv2d layers, not after last two - predictions can be +ve or -ve
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlockUp, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 0),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = 0),
            nn.ReLU(),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _ConvBlockFinal(nn.Module):
    """
    Third block - full scale.
    Six Conv2d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv2d layer
    ReLU after first four Conv2d layers, not after last two - predictions can be +ve or -ve
    """
    def __init__(self, in_channels, out_channels):
        super(_ConvBlockFinal, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
        ]
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class UNet3(nn.Module):
    """
    Define the network. Only input when called is number of data (input) channels.
        -Downsample input to quarter scale and use ConvBlock1.
        -Upsample output of ConvBlock1 to half scale.
        -Downsample input to half scale, concat to output of ConvBlock1; use ConvBLock2.
        -Upsample output of ConvBlock2 to full scale.
        -Concat input to output of ConvBlock2, use ConvBlock3. Output of ConvBlock3 has 8 channels
        -Use final Conv2d layer with kernel size of 1 to go from 8 channels to 1 output channel.
    """
    def __init__(self,data_channels):
        super(UNet3, self).__init__()
        self.convN_1 = _ConvBlockInit(data_channels,32)
        self.convN_2 = _ConvBlockMid(32, 64)
        self.convN_3 = _ConvBlockMid(64,96)
        self.convN_4 = _ConvBlockUp(160,64)
        self.convN_5 = _ConvBlockUp(96,64)
        self.final = _ConvBlockFinal(64,1)

    def forward(self, x):
        convN_1out = self.convN_1(x)
        convN_2out = self.convN_2(convN_1out)
        convN_3out = self.convN_3(convN_2out)
        convN_4out = self.convN_4( torch.cat((F.interpolate(convN_3out,scale_factor=2,mode = 'bilinear'), convN_2out),dim = 1) )
        convN_5out = self.convN_5( torch.cat((F.interpolate(convN_4out,scale_factor=2,mode = 'bilinear'), convN_1out),dim = 1) ) 
        final_out = self.final(convN_5out)
        return final_out
