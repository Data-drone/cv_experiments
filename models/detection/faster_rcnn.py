import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead, RegionProposalNetwork
from torchvision.ops import MultiScaleRoIAlign

# based on https://github.com/pytorch/vision/blob/master/torchvision/models/detection/faster_rcnn.py

__all__ = [
    "fasterrcnn_mobilenetv2_fpn",
]



def fasterrcnn_mobilenetv2_fpn(pretrained=False, progress=True,
                                num_classes=91, pretrained_backbone=True, **kwargs):

    """

    Construct a Faster RCNN model with mobilenetv2 backbone 

    """

    if pretrained:
        pretrained_backbone = False
    
    backbone = torchvision.models.mobilenet_v2(pretrained=pretrained_backbone).features
    backbone.out_channels = 1280

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                        aspect_ratios=((0.5, 1.0, 2.0),))
    
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
                                                    output_size=7,
                                                    sampling_ratio=2)

    model = FasterRCNN(backbone,
                        num_classes=num_classes,
                        rpn_anchor_generator = anchor_generator,
                        box_roi_pool = roi_pooler)

    return model
