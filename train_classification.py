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
from utils import AverageMeter

from torch.optim.lr_scheduler import ReduceLROnPlateau

#TODO
# look at cyclic lr
# look at cyclic momentum
# review
# https://arxiv.org/abs/1803.09820

# load pipelines
from data_pipeline.basic_pipeline import HybridTrainPipe, HybridValPipe
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


train_logger = logging.getLogger(__name__)

wandb.init(project="image_classification_opt")

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
parser.add_argument('--opt', metavar='OPT', default='sgd',
                    choices=['sgd', 'adam', 'adamw'],
                    help='optimiser function')
parser.add_argument('--num-classes', '-nc', metavar='N', default=1000, type=int,
                    help='num classes for classification task (default 1000)')
parser.add_argument('--epochs', '-e', metavar='N', default=10, type=int,
                    help='default num of epochs (default 10)')
parser.add_argument('-j', '--workers', default=6, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
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
parser.add_argument('--opt-level', type=str, default='O0')
parser.add_argument('--keep-batchnorm-fp32', type=str, default=None)
parser.add_argument('--loss-scale', type=str, default=None)

parser.add_argument('--local_rank', default=0, type=int,
        help='Used for multi-process training. Can either be manually set ' +
            'or automatically set by using \'python -m multiproc\'.')


# keep true unless we vary image sizes
cudnn.benchmark = True

args = parser.parse_args()
wandb.config.update(args)

# make apex optional - we aren't using distributed
if args.fp16: #or args.distributed:
    try:
        #from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
        from apex import amp, optimizers
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this script.")


# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    

def train(train_loader, model, criterion, optimizer, epoch):

    model.train()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_list = AverageMeter()
        
    for i, data in enumerate(train_loader):
        
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        train_loader_len = int(train_loader._size / args.batch_size)
        one_cyc_learning_rate(optimizer, epoch, i, train_loader_len)

        input_var = Variable(input)
        target_var = Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        reduced_loss = loss.data

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        optimizer.zero_grad()

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        # hold metrics
        top1.update(to_python_float(prec1), input.size(0) )
        top5.update(to_python_float(prec5), input.size(0) )
        loss_list.update(to_python_float(reduced_loss), input.size(0) )
        
        if i % 20 == 0 and i > 1: 
            stats = {"epoch": epoch, "loss": reduced_loss.item(), "Train Top-1": prec1.item(), 
                        "Train Top-5": prec5.item()}
            print('[{0} / {1}]'.format(i, train_loader_len))
            print(stats)
       
    wandb.log({"epoch": epoch, "train_loss": loss_list.avg, "train_top1": top1.avg,  "train_top5": top5.avg})

        


def validate(val_loader, model, criterion, epoch) -> float:
    
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_list = AverageMeter()

    for i, data in enumerate(val_loader):
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        val_loader_len = int(val_loader._size / args.batch_size)

        target = target.cuda(non_blocking=True)
        input_var = Variable(input)
        target_var = Variable(target)

        # compute output
        with torch.no_grad():
            output = model(input_var)
            loss = criterion(output, target_var)

        # precision
        prec1, prec5 = accuracy(output.data, target, topk=(1,5))

        reduced_loss = loss.data

        # hold metrics
        top1.update( to_python_float(prec1), input.size(0) )
        top5.update( to_python_float(prec5), input.size(0) )
        loss_list.update( to_python_float(reduced_loss), input.size(0) )

        if i % 20 == 0 and i > 1: 
            stats = {"epoch": epoch, "loss": reduced_loss.cpu(), "Val Top-1": prec1.cpu(), 
                        "Val Top-5": prec5.cpu()}
            print('[{0} / {1}]'.format(i, val_loader_len))
            print(stats)

    wandb.log({"epoch": epoch, "val_loss": loss_list.avg, "val_top1": top1.avg,  "val_top5": top5.avg})

    return loss_list.avg

def one_cyc_learning_rate(optimizer, epoch, step, len_epoch):
    # need to add the step back down part
    
    mid_epoch = int((args.epochs)/2)
    
    lr = args.lr
    
    if epoch < mid_epoch:
        lr = lr*float(1 + step + epoch*len_epoch)/(mid_epoch*len_epoch)
        
    if step % 20 == 0 and step > 1:
        wandb.log({"cur_lr": lr})
        print("current learning rate: {0}".format(lr))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # start from the min rate
    # get to the large rate
    
    
# add a custom learning rate optimiser as per the apex reference
def adjust_learning_rate(optimizer, epoch, step, len_epoch):
    """LR schedule that should yield 76% converged accuracy with batch size 256"""
    factor = epoch // 30

    if epoch >= 80:
        factor = factor + 1

    lr = args.lr*(0.1**factor)

    """Warmup"""
    if epoch < 5:
        lr = lr*float(1 + step + epoch*len_epoch)/(5.*len_epoch)
    
    wandb.log({"cur_lr": lr})

    # if(args.local_rank == 0):
    #     print("epoch = {}, step = {}, lr = {}".format(epoch, step, lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    
    
def main():
    
    """
    main training loop function
    """

    # distributed training variable
    args.world_size = 1
    
    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    # TO DO add pretrained handling to local models
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

    # exception for inception v3 as per https://stackoverflow.com/questions/53476305/attributeerror-tuple-object-has-no-attribute-log-softmax#
    if args.arch == 'inception_v3':
        model.aux_logits=False

    model = model.cuda()

    # define loss function (criterion) and optimizer
    #### Edit point for tuning details ####
    criterion = nn.CrossEntropyLoss().cuda()

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)

    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), args.lr,
                                    weight_decay=args.weight_decay)

    #scheduler = MultiStepLR(
    #    optimizer=optimizer, 
    #    milestones=[43, 54], 
    #    gamma=0.1)

    scheduler = ReduceLROnPlateau(optimizer, 'min')
    #scheduler = 

    # scale lr
    # Scale learning rate based on global batch size
    args.lr = args.lr*float(args.batch_size*args.world_size)/256. 

    
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )
    
    traindir = args.data[0]
    valdir= args.data[1]


    # code specific to the basic data_pipeline
    if(args.arch == "inception_v3"):
        crop_size = 299
        val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, 
                            data_dir=traindir, crop=crop_size, dali_cpu=False)
    pipe.build()
    train_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, 
                            data_dir=valdir, crop=crop_size, size=val_size)
    pipe.build()
    val_loader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader") / args.world_size))

    wandb.watch(model)

    for epoch in range(0, args.epochs):

        # train loop
        train(train_loader, model, criterion, optimizer, epoch)
        val_loss = validate(val_loader, model, criterion, epoch)
        
        scheduler.step(val_loss)
        
        # ReduceLROnPlateau doesn't have the get_lr property
        #print('learn rate: {0}'.format(scheduler.get_lr()))
        #wandb.log({"learn_rate": scheduler.get_lr()})

        # for each epoch need to reset
        train_loader.reset()
        val_loader.reset()

if __name__ == '__main__':
    main()