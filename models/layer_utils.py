import torch
import torch.nn as nn
import torch.nn.functional as F

# extra utilities for building neural networks

# from https://github.com/facebookresearch/pycls/blob/master/pycls/models/effnet.py
class Swish(nn.Module):
    """ Swish Activation Function:
     """

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

# from https://github.com/lessw2020/mish/blob/master/mish.py
class Mish(nn.Module):
    """ Mish Activation Function """

    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * ( torch.tanh(F.softplus(x)) )