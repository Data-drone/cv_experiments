# resnet but with optimisations from https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

import logging

model_logger = logging.getLogger(__name__ + '.models.opt_resnet')