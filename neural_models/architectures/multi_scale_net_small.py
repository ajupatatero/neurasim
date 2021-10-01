"""
3 level multiscale network

Inputs are shape (batch, channels, height, width)
Outputs are shape (batch,1, height, width)

The number of input (data) channels is selected when the model is created.
the number of output (target) channels is fixed at 1, although this could be changed in the future.

The data can be any size (i.e. height and width), although for best results the height and width should
be divisble by four.

The model can be trained on data of a given size (H and W) and then used on data of any other size,
although the best results so far have been obtained with test data of similar size to the training data

"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBlock1(nn.Module):
    """
    First block - quarter scale.
    Four Conv2d layers, all with kernel_size 3 and padding of 1 (padding ensures output size is same as input size)
    Optional dropout before final Conv2d layer
    ReLU after first two Conv2d layers, not after last two - predictions can be +ve or -ve
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels, out_channels,dropout=False):
        super(_ConvBlock1, self).__init__()
        layers = [
            nn.ReplicationPad2d(1),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, mid2_channels, kernel_size=3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels,mid1_channels,kernel_size = 3),
        ]

        layers.append(nn.ReplicationPad2d(1))
        layers.append(nn.Conv2d(mid1_channels, out_channels, kernel_size = 3))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlock2(nn.Module):
    """
    Second block - half scale.
    Six Conv2d layers. First one kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv2d layer
    ReLU after first four Conv2d layers, not after last two - predictions can be +ve or -ve
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels,mid3_channels, out_channels,dropout=False):
        super(_ConvBlock2, self).__init__()
        layers = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, mid2_channels, kernel_size=3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels,mid3_channels,kernel_size = 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid3_channels,mid2_channels,kernel_size = 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels,mid1_channels,kernel_size = 3),
        ]
        layers.append(nn.ReplicationPad2d(1))
        layers.append(nn.Conv2d(mid1_channels, out_channels, kernel_size = 3))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _ConvBlock3(nn.Module):
    """
    Third block - full scale.
    Six Conv2d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv2d layer
    ReLU after first four Conv2d layers, not after last two - predictions can be +ve or -ve
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels,mid3_channels, out_channels,dropout=False):
        super(_ConvBlock3, self).__init__()
        layers = [
            nn.ReplicationPad2d(2),
            nn.Conv2d(in_channels, mid1_channels, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid1_channels, mid2_channels, kernel_size=3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels,mid3_channels,kernel_size = 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid3_channels,mid2_channels,kernel_size = 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels,mid1_channels,kernel_size = 3),
        ]
        layers.append(nn.ReplicationPad2d(2))
        layers.append(nn.Conv2d(mid1_channels, out_channels, kernel_size = 5))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class MultiScaleNetSmall(nn.Module):
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
        super(MultiScaleNetSmall, self).__init__()
        self.convN_4 = _ConvBlock1(data_channels, 32,64,1)
        self.convN_2 = _ConvBlock2(data_channels+1, 16,32,64,1)
        self.convN_1 = _ConvBlock3(data_channels+1, 16,32,64,8)
        self.final = nn.Conv2d(8,1, kernel_size = 1)

    def forward(self, x):
        align = False
        quarter_size = [int(i*0.25) for i in list(x.size()[2:])]
        half_size = [int(i*0.5) for i in list(x.size()[2:])]

        not_recording = True

        if not_recording:
            convN_4out = self.convN_4(F.interpolate(x,(quarter_size),mode = 'bilinear',align_corners=align))
            convN_2out = self.convN_2( torch.cat((F.interpolate(x,(half_size),mode = 'bilinear',align_corners=align),
                                         F.interpolate(convN_4out,(half_size),mode = 'bilinear',align_corners=align)),dim = 1) )
            convN_1out = self.convN_1( torch.cat((F.interpolate(x,(x.size()[2:]),mode = 'bilinear',align_corners=align),
                                         F.interpolate(convN_2out,(x.size()[2:]),mode = 'bilinear',align_corners=align)),dim = 1) )

            final_out = torch.cat((self.final(convN_1out),\
                    F.interpolate(convN_2out,(x.size()[2:]),mode = 'bilinear',align_corners=align),\
                    F.interpolate(F.interpolate(convN_4out,(half_size),mode = 'bilinear',align_corners=align),\
                    (x.size()[2:]),mode = 'bilinear',align_corners=align)),dim=1)

        else:

            interpol_4 = F.interpolate(x,(quarter_size),mode = 'bilinear',align_corners=align)
            convN_4out = self.convN_4(interpol_4)
            interpol_2 = F.interpolate(x,(half_size),mode = 'bilinear',align_corners=align)
            interpol_4_b = F.interpolate(convN_4out,(half_size),mode = 'bilinear',align_corners=align)
            concatenated_2 = torch.cat((interpol_2,interpol_4_b),dim=1)
            convN_2out = self.convN_2(concatenated_2)
            interpol_1 = F.interpolate(x,(x.size()[2:]),mode = 'bilinear',align_corners=align)
            interpol_2_b = F.interpolate(convN_2out,(x.size()[2:]),mode = 'bilinear',align_corners=align)
            concatenated_1 = torch.cat((interpol_1, interpol_2_b), dim =1)
            convN_1out = self.convN_1(concatenated_1)
            final_big = self.final(convN_1out)
            interpolate_n2 = F.interpolate(convN_2out,(x.size()[2:]),mode = 'bilinear',align_corners=align)
            interpolate_n4 = F.interpolate(F.interpolate(convN_4out,(half_size),mode = 'bilinear',align_corners=align),(x.size()[2:]),mode = 'bilinear',align_corners=align)
            concat_fin = torch.cat((final_big, interpolate_n2, interpolate_n4), dim =1)
            final_out = concat_fin

        #return final_out
        return final_out[:,0].unsqueeze(1), final_out
