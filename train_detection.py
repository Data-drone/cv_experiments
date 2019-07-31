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
import torchvision.models.detection as models
import wandb
import logging
import models as local_models
from utils import AverageMeter

# load pipelines
from data_pipeline.coco_pipeline import COCOTrainPipeline, COCOValPipeline, CocoSimple
try:
    from nvidia.dali.plugin.pytorch import DALIGenericIterator
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


train_logger = logging.getLogger(__name__)

#wandb.init(project="image_detection")

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

local_model_names = sorted(name for name in local_models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(local_models.__dict__[name]))

valid_models = model_names + local_model_names

parser = argparse.ArgumentParser(description="PyTorch Model Training")

parser.add_argument('data', metavar='DIR', nargs='*',
                    help='paths to dataset and annotations')
parser.add_argument('--arch', '-a', metavar='ARCH', default='',
                    choices=valid_models,
                    help='model architecture: | {0} (default: resnet18)'.format(valid_models))
parser.add_argument('--opt', metavar='OPT', default='sgd',
                    choices=['sgd', 'adam'],
                    help='optimiser function')
parser.add_argument('--num-classes', '-nc', metavar='N', default=80, type=int,
                    help='num classes for detection task (default 8080)')
parser.add_argument('--epochs', '-e', metavar='N', default=10, type=int,
                    help='default num of epochs (default 10)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N', help='mini-batch size (default: 32)')
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
parser.add_argument('--static-loss-scale', type=float, default=1,
                    help='Static loss scale, positive power of 2 values can improve fp16 convergence.')
parser.add_argument('--dynamic-loss-scale', action='store_true',
                    help='Use dynamic loss scaling.  If supplied, this argument supersedes ' +
                    '--static-loss-scale.')
parser.add_argument('--local_rank', default=0, type=int,
        help='Used for multi-process training. Can either be manually set ' +
            'or automatically set by using \'python -m multiproc\'.')

# keep true unless we vary image sizes
cudnn.benchmark = True

args = parser.parse_args()
#wandb.config.update(args)

# make apex optional - we aren't using distributed
if args.fp16: #or args.distributed:
    try:
        #from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this script.")


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]


def train(train_loader, model, optimizer, epoch):
    
    model.train()

    for i, data in enumerate(train_loader):

        print(type(data))
        print(data)

        images = data[0]
        boxes = data[1]
        #labels = data[2]

        losses = model(images=images, boxes=boxes)

        






def validate(val_loader):
    
    model.eval()

    for i, data in enumerate(val_loader):

        pass

#def main():

"""
main training loop function
"""

# distributed training variable
args.world_size = 1

if args.fp16:
    assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

if args.static_loss_scale != 1.0:
    if not args.fp16:
        print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")

# TO DO add pretrained handling to local models
# need to revise the way we do the model creation for faster rcnn

if args.pretrained:
    print("=> using pre-trained model '{}'".format(args.arch))
    if args.arch in model_names:
        model = models.__dict__[args.arch](pretrained=True)
    elif args.arch in local_model_names:
        model = local_models.__dict__[args.arch](pretrained=True)
else:
    print("=> creating new model '{}'".format(args.arch))
    if args.arch in model_names:
        model = models.__dict__[args.arch](pretrained=False)
    elif args.arch in local_model_names:
        model = local_models.__dict__[args.arch](pretrained=False)

model = model.cuda()

if args.fp16:       
    model = network_to_half(model)

if args.opt == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

if args.opt == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                weight_decay=args.weight_decay)


if args.fp16:
    optimizer = FP16_Optimizer(optimizer,
                                static_loss_scale=args.static_loss_scale,
                                dynamic_loss_scale=args.dynamic_loss_scale)

# can I add a scheduler here? - only for sgd?

traindir = args.data[0]
valdir= args.data[1]
annotationsdir = args.data[2]

pipe_test = CocoSimple(batch_size = args.batch_size, num_threads = args.workers, device_id = args.local_rank,
file_root = traindir, annotations_file = annotationsdir, num_gpus=1)

train_pipe = COCOTrainPipeline(batch_size = args.batch_size, num_threads = args.workers,
                device_id=args.local_rank, 
                file_root = traindir, annotations_file = annotationsdir)


# size has been hard coded for now 
# need to restructure the way that it is called and how it is initiated
# as per the SSD detector in the demo notes

#train_loader = DALIGenericIterator(train_pipe, ["images", "boxes", "labels"],
#                            118287, stop_at_epoch=False)

# do we need two annotations? the size has been hardcoded for now
#val_pipe = COCOValPipeline(batch_size = args.batch_size, num_threads = args.workers,
#                device_id=args.local_rank, 
#                file_root = valdir, annotations_file = annotationsdir)

#val_loader = DALIGenericIterator(val_pipe, ["images", "boxes", "labels"],
#                            5000 , stop_at_epoch=False)

#wandb.watch(model)

#for epoch in range(0, args.epochs):

# is this all we need?
#    train(train_loader, model, optimizer, epoch)
#    validate(val_loader, model, optimizer, epoch)