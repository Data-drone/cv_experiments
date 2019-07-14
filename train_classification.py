import argparse
import os
import shutil
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
import wandb
import logging
import models as local_models

# load pipelines
from data_pipeline.basic_pipeline import HybridTrainPipe, HybridValPipe

train_logger = logging.getLogger(__name__)

wandb.init(project="image_classification")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

local_model_names = sorted(name for name in local_models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(local_models.__dict__[name]))

valid_models = model_names + local_model_names

parser = argparse.ArgumentParser(description="PyTorch Model Training")

# Parameterise
## data / architecture
## target accuracy?
parser.add_argument('data', metavar='DIR', nargs='*',
                    help='paths to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH',
                    choices=valid_models,
                    help='model architecture: | {0} (default: resnet18)'.format(valid_models))
parser.add_argument('--num-classes', '-nc', metavar='N', default=1000, type=int,
                    help='num classes for classification task (default 1000)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--fp16', action='store_true',
                    help='Run model fp16 mode.')

# keep true unless we vary image sizes
cudnn.benchmark = True

args = parser.parse_args()
wandb.config.update(args)

