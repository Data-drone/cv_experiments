import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

# wide resnet from scratch
# https://arxiv.org/pdf/1605.07146.pdf?

__all__ = ['wide_resnet22_1',
            'wide_resnet22_2',
            'wide_resnet22_4',
            'wide_resnet22_8',
            'wide_resnet28_10']

model_url = {
    'wide_resnet22_1': '',
    'wide_resnet22_2': '',
    'wide_resnet22_4': '',
    'wide_resnet22_8': '',
    'wide_resnet28_10': ''
}

# TODO
# fix layer sizing
# flex k

class BasicBlock(nn.Module):
    
    def __init__(self, insize, filters, downsample=None):
        super(BasicBlock, self).__init__()
                
        self.bn1 = nn.BatchNorm2d(insize)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=insize, out_channels=filters, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.downsample = downsample

        self.dropout = nn.Dropout(p=0.2)
        
    def forward(self,x):
        identity = x
        
        out = self.bn1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.dropout(out)
        
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # when we change size we need to downsample I guess?
        out += identity
        
        return out


class Bottleneck(nn.Module):

    def __init__(self, insize, filters, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(insize)
        self.act = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels=insize, out_channels=filters, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        self.bn2 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=1, stride=1,
                     padding=0, groups=1, bias=False, dilation=1)

        self.bn3 = nn.BatchNorm2d(filters)
        self.conv3 = nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, stride=1,
                     padding=1, groups=1, bias=False, dilation=1)
        
        self.downsample = downsample

        self.dropout = nn.Dropout(p=0.2)

    def forward(self,x):
        identity = x
        
        out = self.bn1(x)
        out = self.act(out)
        out = self.conv1(out)

        out = self.dropout(out)
        
        out = self.bn2(out)
        out = self.act(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.act(out)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        # when we change size we need to downsample I guess?
        out += identity
        
        return out



class WResNet(nn.Module):
    """
    Wide Resnet Class - uses three groups
    """

    def __init__(self, block, layers=[2,2,2], k=4, num_classes=1000):
        super(WResNet, self).__init__()
        
        self.bn1 = nn.BatchNorm2d(3) # norm variable in torchvision
        self.conv1 = nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.act = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        #### first part
        
        self.group1 = self._make_layer(block, 16, 16*k, layers[0])
        self.group2 = self._make_layer(block, 16*k, 32*k, layers[1])
        
        #### second part
        
        self.group3 = self._make_layer(block, 32*k, 64*k, layers[2])
        #self.group4 = self._make_layer(block, 64*k, 128*k, layers[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64*k, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # need see what to do here
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            
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
        x = self.act(x)
        x = self.conv1(x)
        # does changing order change this?
        x = self.maxpool(x)
        
        x = self.group1(x)
        x = self.group2(x)
        
        x = self.group3(x)
        #x = self.group4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x


def _wide_resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = WResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_url[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model

def wide_resnet22_1(pretrained=False, progress=True, **kwargs):
    r"""
     '"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>'
     This model is based on resnet with the extra width parameters used in the paper
     is uses a Basic B(3,3) block.
     Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _wide_resnet('wide_resnet22_1', BasicBlock, [3,3,3],
                        pretrained, progress, k=1, **kwargs)

def wide_resnet22_2(pretrained=False, progress=True, **kwargs):
    r"""
     '"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>'
     This model is based on resnet with the extra width parameters used in the paper
     is uses a Basic B(3,3) block with resnet 22 replication
     Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _wide_resnet('wide_resnet22_2', BasicBlock, [3,3,3],
                        pretrained, progress, k=2, **kwargs)                     

def wide_resnet22_4(pretrained=False, progress=True, **kwargs):
    r"""
     '"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>'
     This model is based on resnet with the extra width parameters used in the paper
     is uses a Basic B(3,3) block with resnet 22 replication
     Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _wide_resnet('wide_resnet22_4', BasicBlock, [3,3,3],
                        pretrained, progress, k=4, **kwargs)                     

def wide_resnet22_8(pretrained=False, progress=True, **kwargs):
    r"""
     '"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>'
     This model is based on resnet with the extra width parameters used in the paper
     is uses a Basic B(3,3) block with resnet 22 replication
     Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _wide_resnet('wide_resnet22_8', BasicBlock, [3,3,3],
                        pretrained, progress, k=4, **kwargs)                     


# wide_resnet28_10
def wide_resnet28_10(pretrained=False, progress=True, **kwargs):
    r"""
     '"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>'
     This model is based on resnet with the extra width parameters used in the paper
     is uses a Basic B(3,3) block with resnet 22 replication
     Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """

    return _wide_resnet('wide_resnet28_10', BasicBlock, [4,4,4],
                        pretrained, progress, k=10, **kwargs)                     
