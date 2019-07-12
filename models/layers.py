import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

layer_logger = logging.getLogger(__name__ + '.models.layers') 

class ConvBN(nn.Module):
    #convolutional layer then Batchnorm
    def __init__(self, ch_in, ch_out, kernel_size = 3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(ch_out, momentum=0.01)

    def forward(self, x):
        output = F.leaky_relu(self.bn(self.conv(x)), negative_slope=0.1) 
        # debugging
        layer_logger.info("ConvBN layer output shape: {0}".format(output.shape))
        return output


class MaxPoolStride1(nn.Module):
    def __init__(self):
        super(MaxPoolStride1, self).__init__()

    def forward(self, x):
        x = F.max_pool2d(F.pad(x, (0,1,0,1), mode='constant'), 2, stride=1)
        layer_logger.info("MaxPoolStride1 layer output shape: {0}".format(x.shape))
        return x

class ConvPool(nn.Module):
    def __init__(self, ch_out):
        super().__init__()
        ch_in = int(ch_out / 2)
        self.conv1 = ConvBN(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        layer_logger.info("ConvPool layer output shape: {0}".format(out.shape))
        return out

    
class DarknetBlock(nn.Module):
    #The basic blocs.   
    def __init__(self, ch_in):
        super().__init__()
        ch_hid = ch_in//2
        self.conv1 = ConvBN(ch_in, ch_hid, kernel_size=1, stride=1, padding=0)
        self.conv2 = ConvBN(ch_hid, ch_in, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out + x


