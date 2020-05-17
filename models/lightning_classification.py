from models.wide_resnet import _wide_resnet
from pytorch_lightning.core import LightningModule

## lightning wrapper around wide resnet
## initiate then wrap the model into a LightningModule
# Need to test

class WResNet_pl(LightningModule):

    def __init__(self, arch, block, layers, pretrained, progress, **kwargs):
        super().__init__()

        self.model = _wide_resnet(arch, block, layers, pretrained, progress, **kwargs)

    def forward(self, x):

        return self.model(x)


    #TODO - add the other methods or can make a generic wrapper class?