# Darknet backbone as used in yolo network
# follows torchvision model def format

import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import MaxPoolStride1, ConvBN, DarknetBlock, ConvPool

import logging

models_logger = logging.getLogger(__name__ + '.models.darknet')

__all__ = ['TinyDarknet', 'tinydarknetv3']

class TinyDarknet(nn.Module):
    # replicates tiny yolov3 backbone
    def __init__(self, num_blocks, num_classes=1000):
        super().__init__()
        self.conv1 = ConvBN(3, 16, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.layer1 = self.make_tiny_yolo_stack(32, num_blocks, stride=1)
        self.eight_layer = ConvBN(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv1_1 = ConvBN(256, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool1 = MaxPoolStride1()
        self.conv2 = ConvBN(512, 1024, kernel_size=3, stride=1, padding=1) # hack
        self.conv3 = ConvBN(1024, 256, kernel_size=1, stride=1, padding=1)
        self.conv4 = ConvBN(256, 512, kernel_size=3, stride=1, padding=1)
        self.linear = nn.Linear(512, num_classes)
        
        
    def make_tiny_yolo_stack(self, ch_in, num_blocks, stride=1):
        layers = []
        for i in range(num_blocks):
            layers.append(ConvPool(ch_in))
            ch_in = int(2*ch_in)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.maxpool(out)
        out = self.layer1(out)
        self.eight_out = self.eight_layer(out)
        out = self.maxpool(self.eight_out)
        out = self.conv1_1(out) # added in later to fix error in structure
        out = self.maxpool1(out)
        out = self.conv2(out)
        self.conv_3_out = self.conv3(out)
        out = self.conv4(self.conv_3_out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return F.log_softmax(self.linear(out))

def tinydarknetv3(pretrained=False, **kwargs):
    return TinyDarknet(num_blocks=3,**kwargs)