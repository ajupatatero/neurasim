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


class _ConvBlock3(nn.Module):
    """
    Third block - full scale.
    Six Conv2d layers. First and last kernel size 5, padding 2, remainder kernel size 3 padding 1.
    Optional dropout before final Conv2d layer
    ReLU after first four Conv2d layers, not after last two - predictions can be +ve or -ve
    """
    def __init__(self, in_channels, mid1_channels, mid2_channels,mid3_channels, mid4_channels,out_channels,dropout=False):
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
            nn.Conv2d(mid3_channels,mid4_channels,kernel_size = 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid4_channels,mid3_channels,kernel_size = 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid3_channels,mid2_channels,kernel_size = 3),
            nn.ReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(mid2_channels,mid1_channels,kernel_size = 3),
            nn.ReLU(),
        ]
        layers.append(nn.ReplicationPad2d(2))
        layers.append(nn.Conv2d(mid1_channels, out_channels, kernel_size = 5))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class MultiScaleNetMonoBig(nn.Module):
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
        super(MultiScaleNetMonoBig, self).__init__()
        #self.convN_4 = _ConvBlock1(data_channels, 32,64,1)
        #self.convN_2 = _ConvBlock2(data_channels+1, 32,64,128,1)
        self.convN_1 = _ConvBlock3(data_channels, 32,32,128,128,8)
        self.final = nn.Conv2d(8,1, kernel_size = 1)

    def forward(self, x):

        align = False
        print_detail = False

        if print_detail:
            event_1 = torch.cuda.Event(enable_timing=True)
            event_2 = torch.cuda.Event(enable_timing=True)
            event_3 = torch.cuda.Event(enable_timing=True)
            #event_4 = torch.cuda.Event(enable_timing=True)

            # Start recording
            #event_1.record()
            #torch.cuda.synchronize()
          
            #interpol = F.interpolate(x,(x.size()[2:]),mode = 'bilinear',align_corners=align)

            event_1.record()
            torch.cuda.synchronize()

            convN_1out = self.convN_1(x)

            event_2.record()
            torch.cuda.synchronize()

            final_out = self.final(convN_1out)

            event_3.record()
            torch.cuda.synchronize()

            elapsed_time_1 = event_1.elapsed_time(event_2)
            elapsed_time_2 = event_2.elapsed_time(event_3)
            #elapsed_time_3 = event_3.elapsed_time(event_4)
            elapsed_time_total = event_1.elapsed_time(event_3)

            print("Elapsed Total Time: ",elapsed_time_total)
            print("Step 1: ",elapsed_time_1)
            print("Step 2: ",elapsed_time_2)
            #print("Step 3: ",elapsed_time_3)

        else:

            #convN_1out = self.convN_1(F.interpolate(x,(x.size()[2:]),mode = 'bilinear',align_corners=align))
            #final_out = self.final(convN_1out)
            convN_1out = self.convN_1(x)
            final_out = self.final(convN_1out)

        return final_out
