import torch
import torch.nn as nn

# wide resnet from scratch
# https://arxiv.org/pdf/1605.07146.pdf?

__all__ = ['wide_resnet18']

model_url = {
    'wide_resnet18': ''
}

class BasicBlock(nn.Module):
    
    def __init__(self, insize, filters, downsample=None):
        super(BasicBlock, self).__init__()
                
        self.bn1 = nn.BatchNorm2d(insize)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=insize, out_channels=filters, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.downsample = downsample
        
    def forward(self,x):
        identity = x
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # when we change size we need to downsample I guess?
        out += identity
        
        return out



class WResNet(nn.Module):

    def __init__(self, block, layers=[2,2,2,2], k=4, num_classes=1000):
        super(TestModule, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(3) # norm variable in torchvision
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #### first part
        
        self.group1 = self._make_layer(block, 16*k, 16*k, layers[0])
        self.group2 = self._make_layer(block, 16*k, 32*k, layers[1])
        
        #### second part
        
        self.group3 = self._make_layer(block, 32*k, 64*k, layers[2])
        self.group4 = self._make_layer(block, 64*k, 128*k, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128*k, num_classes)
            
    def _make_layer(self, block, in_planes, out_planes, num_blocks, stride=1, dilate=False):
        downsample = None
        
        downsample_conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1)
        downsample_norm = nn.BatchNorm2d(out_planes)
        
        if in_planes != out_planes:
            #downsample = nn.Sequential(
            #    conv1x1(in_planes, out_planes, stride),
            #    nn.BatchNorm2d(out_planes),
            #)
            downsample = nn.Sequential(
                downsample_conv,
                #conv1x1(in_planes, out_planes, stride),
                downsample_norm
            )
        
        layers = []
        layers.append(block(insize=in_planes, filters=out_planes, downsample=downsample))
        
        for _ in range(1, num_blocks):
            layers.append(block(insize=out_planes, filters=out_planes))
        
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        # does changing order change this?
        x = self.maxpool(x)
        
        x = self.group1(x)
        x = self.group2(x)
        
        x = self.group3(x)
        x = self.group4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def _wide_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = WResNet(block, layers, **kwargs):
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def wide_resnet18(pretrained=False, progress=True, **kwargs):
    r"""
     '"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>'
     This model is based on resnet with the extra width parameters used in the paper
     is uses a Basic B(3,3) block.
     Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _wide_resnet('wide_resnet18', BasicBlock, [2,2,2,2],
                        pretrained, progress, **kwargs)