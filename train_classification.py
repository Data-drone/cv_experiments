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
import optimisers as local_optimisers
from utils import AverageMeter
from lr_schedulers.onecyclelr import OneCycleLR

from torch.optim.lr_scheduler import ReduceLROnPlateau

from loss.label_smoothing import LabelSmoothing
from models.layer_utils import Swish, Mish


#TODO
# look at cyclic lr - added one cycle
# look at cyclic momentum
# review
# https://arxiv.org/abs/1803.09820

# load pipelines

######

DATA_BACKEND_CHOICES = ['dali-cpu', 'dali-gpu']

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
                    choices=['sgd', 'adam', 'adamw', 'radam', 'ranger'],
                    help='optimiser function')
parser.add_argument('--data-backend', metavar='BACKEND', default='dali-cpu',
                    choices=DATA_BACKEND_CHOICES)
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
parser.add_argument('--label-smoothing', type=float, default=0.0,
                    help='add in label smoothing')

### fp 16 arguments
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
parser.add_argument('--opt-level', type=str, default='O0')

### change activation function
ACT_FUNC = ['relu','swish', 'mish']
parser.add_argument('--act-func', default='relu',
                    choices=ACT_FUNC)

# keep true unless we vary image sizes
cudnn.benchmark = True

args = parser.parse_args()
wandb.config.update(args)

# make apex optional - we aren't using distributed
if args.fp16: #or args.distributed:
    try:
        from apex.parallel import DistributedDataParallel as DDP
        from apex.fp16_utils import *
        from apex import amp, optimizers
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this script.")

args.distributed = False
if 'WORLD_SIZE' in os.environ:
    args.distributed = int(os.environ['WORLD_SIZE']) > 1

#######

# item() is a recent addition, so this helps with backward compatibility.
def to_python_float(t):
    if hasattr(t, 'item'):
        return t.item()
    else:
        return t[0]

# tensor reduce to gather from different gpus
def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= args.world_size
    return rt

def accuracy(output, target, topk=(1,)) -> list:
    """Computes the precision@k for the specified values of k"""
    # should we be calling this accuracy?
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


def train(train_loader, model, criterion, optimizer, epoch, scheduler):
    # Main train loop
    # loads batches from the train loader
    # and loops through 1 epoch


    model.train()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_list = AverageMeter()
        
    for i, data in enumerate(train_loader):
        
        input = data[0]["data"]
        target = data[0]["label"].squeeze().cuda().long()
        train_loader_len = int(train_loader._size / args.batch_size)
        #one_cyc_learning_rate(optimizer, epoch, i, train_loader_len)

        input_var = Variable(input)
        target_var = Variable(target)
                    
        output = model(input_var)
        loss = criterion(output, target_var)

        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

       
        optimizer.zero_grad()

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
            
        optimizer.step()
        #scheduler.step()

        torch.cuda.synchronize()

        # hold metrics
        top1.update(to_python_float(prec1), input.size(0) )
        top5.update(to_python_float(prec5), input.size(0) )
        loss_list.update(to_python_float(reduced_loss), input.size(0) )
        
        if i % 20 == 0 and i > 1: 
            stats = {"epoch": epoch, "loss": reduced_loss.item(), "Train Top-1": prec1.item(), 
                        "Train Top-5": prec5.item(), "cur_lr": scheduler.current_lr}
            progress = '[{0} / {1}]'.format(i, train_loader_len)
            print("{0} - {1}".format(progress, stats))
       
    wandb.log({"epoch": epoch, "train_loss": loss_list.avg, "train_top1": top1.avg,  "train_top5": top5.avg, "cur_lr": scheduler.current_lr})

        


def validate(val_loader, model, criterion, epoch) -> dict:
    
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

        if args.distributed:
            reduced_loss = reduce_tensor(loss.data)
            prec1 = reduce_tensor(prec1)
            prec5 = reduce_tensor(prec5)
        else:
            reduced_loss = loss.data

        # hold metrics
        top1.update( to_python_float(prec1), input.size(0) )
        top5.update( to_python_float(prec5), input.size(0) )
        loss_list.update( to_python_float(reduced_loss), input.size(0) )

        if i % 20 == 0 and i > 1: 
            stats = {"epoch": epoch, "loss": reduced_loss.item(), "Val Top-1": prec1.item(), 
                        "Val Top-5": prec5.item()}
            progress = '[{0} / {1}]'.format(i, val_loader_len)
            print("{0} - {1}".format(progress, stats))
    wandb.log({"epoch": epoch, "val_loss": loss_list.avg, "val_top1": top1.avg,  "val_top5": top5.avg})

    return {'val_loss': loss_list.avg, 'val_top1': top1.avg}
    
    
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


def save_checkpoint(state, is_best, fp_16, model, folder_name='log_models', onnx=False):
    # saves checkpints into the log folder
    # saves the best format one into an onnx file as well - for tensorRT conversion + deployment
    # maybe move crop size into state?
    
    filename = 'img_class_' + str(state['arch']) + '_' + str(state['num_classes']) + 'c'
    filenamepath = os.path.join(folder_name, filename + '_chkpnt.pth.tar')

    # I have altered state need to check this func
    torch.save(state, filenamepath)
    
    if is_best:
        shutil.copyfile(filenamepath, os.path.join(folder_name, filename + '_best.pth.tar'))
        
        if onnx:
            # specify inputs and outputs for onnx
            dummy_input = torch.zeros(1, 3, state['resize'][0], state['resize'][1]).to('cuda')
            inputs = ['images']
            outputs = ['scores']
            dynamic_axes = {'images': {0: 'batch'}, 'scores': {0: 'batch'}}
            
            onnx_name = os.path.join(folder_name, filename + '.onnx')

            if fp_16:
                with amp.disable_casts():
            
                    torch.onnx.export(model, dummy_input, onnx_name, verbose=True, \
                                    input_names=inputs, output_names=outputs)
                
            else:
                torch.onnx.export(model, dummy_input, onnx_name, verbose=True, \
                                    input_names=inputs, output_names=outputs)
                


    
def main():
    
    """
    main training loop function
    """

    # distributed training variable
    args.gpu = 1
    args.world_size = 1
    
    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."

    ### distributed deep learn parameters
    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        torch.distributed.init_process_group(backend='nccl',
                                             init_method='env://')
        args.world_size = torch.distributed.get_world_size()

    args.total_batch_size = args.world_size * args.batch_size

    if args.static_loss_scale != 1.0:
        if not args.fp16:
            print("Warning:  if --fp16 is not used, static_loss_scale will be ignored.")


    if args.act_func == 'swish':
        act_funct = Swish()
    elif args.act_func == 'mish':
        act_funct = Mish()
    elif args.act_func == 'relu':
        act_funct = nn.ReLU(inplace=True)

    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        if args.arch in model_names:
            model = models.__dict__[args.arch](pretrained=True)
        elif args.arch in local_model_names:
            model = local_models.__dict__[args.arch](pretrained=True, 
                                        activation=act_funct)
    else:
        print("=> creating new model '{}'".format(args.arch))
        if args.arch in model_names:
            model = models.__dict__[args.arch](pretrained=False)
        elif args.arch in local_model_names:
            model = local_models.__dict__[args.arch](pretrained=False,
                                                    activation=act_funct)

    # exception for inception v3 as per https://stackoverflow.com/questions/53476305/attributeerror-tuple-object-has-no-attribute-log-softmax#
    if args.arch == 'inception_v3':
        model.aux_logits=False

    print(model)

    model = model.cuda()
    
    # define loss function (criterion) and optimizer
    #### Edit point for tuning details ####
    criterion = nn.CrossEntropyLoss().cuda()
    
    if args.label_smoothing > 0:
        criterion = LabelSmoothing(args.label_smoothing).cuda()


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
        
    if args.opt == 'radam':
        optimizer = local_optimisers.RAdam(model.parameters(), args.lr,
                                          betas=(0.9, 0.999), eps=1e-8,
                                           weight_decay=args.weight_decay)

    if args.opt == 'ranger':
        optimizer = local_optimisers.Ranger(model.parameters(), args.lr,
                                            weight_decay=args.weight_decay)
    
    
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      loss_scale="dynamic" if args.dynamic_loss_scale else args.static_loss_scale
                                      )
    
    if args.distributed:
        # shared param/delay all reduce turns off bucketing in DDP, for lower latency runs this can improve perf
        # for the older version of APEX please use shared_param, for newer one it is delay_allreduce
        model = DDP(model, delay_allreduce=True)


    traindir = args.data[0]
    valdir= args.data[1]


    # code specific to the basic data_pipeline
    if(args.arch == "inception_v3"):
        crop_size = 299
        val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256

    # instantiate the dali pipes
    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        world_size = 1

    print('local rank: {0}'.format(local_rank))

    if args.data_backend == 'dali-cpu':
        tpipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, 
                            data_dir=traindir, crop=crop_size, dali_cpu=True)
    else:
        tpipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, 
                            data_dir=traindir, crop=crop_size, dali_cpu=False)

    tpipe.build()
    train_loader = DALIClassificationIterator(tpipe, size=int(tpipe.epoch_size("Reader") / world_size))

    vpipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, 
                            data_dir=valdir, crop=crop_size, size=val_size)
    vpipe.build()
    val_loader = DALIClassificationIterator(vpipe, size=int(vpipe.epoch_size("Reader") / world_size))
    
    # may need to revisit
    train_loader_len = int(train_loader._size / args.batch_size)*args.epochs
    scheduler = OneCycleLR(optimizer, num_steps=train_loader_len, lr_range=(args.lr/10, args.lr))

    wandb.watch(model)

    best_top1 = 0
    
    for epoch in range(0, args.epochs):

        # train loop
        train(train_loader, model, criterion, optimizer, epoch, scheduler)
        val_dict = validate(val_loader, model, criterion, epoch)

        prec1 = val_dict['val_top1']
        
        is_best = prec1 > best_top1
        best_top1 = max(val_dict['val_top1'], best_top1)

        save_checkpoint({'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'val_loss': val_dict['val_loss'],
                        'optimizer': optimizer.state_dict(),
                        'num_classes': args.num_classes,
                        'resize': (crop_size, crop_size)}, 
                        is_best, args.fp16, model)

        # step scheduler
        scheduler.step()

        # for each epoch need to reset
        train_loader.reset()
        val_loader.reset()
        
        
        
if __name__ == '__main__':
    main()