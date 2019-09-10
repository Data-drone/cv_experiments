#### train classification with ignite

from argparse import ArgumentParser
from typing import Sequence
from math import ceil

import models as local_models

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim
#import optimisers as  
import torch.utils.data
import torchvision.models as models
from torchvision.datasets import ImageFolder
import optimisers as local_optimisers

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
#from ignite.engine import create_supervised_dali_trainer, create_supervised_dali_evaluator
from ignite.metrics import Accuracy, Loss, TopKCategoricalAccuracy, RunningAverage
from lr_schedulers.onecyclelr import OneCycleLR
from ignite.handlers import EarlyStopping

from tqdm import tqdm

## logging
import wandb

## Look into
# https://github.com/pytorch/ignite/commit/f02a91a28153e08092dbb625472e0211f642923d
# on integrating dali with ignite

# load pipelines
from data_pipeline.basic_pipeline import HybridTrainPipe, HybridValPipe
try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator, DALIGenericIterator
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

local_model_names = sorted(name for name in local_models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(local_models.__dict__[name]))

valid_models = model_names + local_model_names

###### DALI Specific ######
# from https://github.com/chicham/ignite/tree/feature/dali
def _pipelines_sizes(pipes):
    for p in pipes:
        p.build()
        keys = list(p.epoch_size().keys())
        if len(keys) > 0:
            for k in keys:
                yield p.epoch_size(k)
        else:
            yield len(p)

class DALILoader(DALIGenericIterator):
    """
    Class to make a `DALIGenericIterator` because `ProgressBar` wants an object with a
    `__len__` method. Also the `ProgressBar` is updated by step of 1 !
    """

    def __init__(self, pipelines, output_map, auto_reset=False, stop_at_epoch=False):
        if not isinstance(pipelines, Sequence):
            pipelines = [pipelines]
        size = sum(_pipelines_sizes(pipelines))
        super().__init__(pipelines, output_map, size, auto_reset, stop_at_epoch)
        self.batch_size = pipelines[0].batch_size

    def __len__(self):
        return int(ceil(self._size / self.batch_size))

from ignite.utils import convert_tensor

def prepare_dali_batch(batch, device=None, non_blocking=False):
    # extract out of data pipeline
    x = batch[0]["data"]
    y = batch[0]["label"]
    # convert
    y = y.squeeze().long()
    
    return (convert_tensor(x, device=device, non_blocking=non_blocking),
            convert_tensor(y, device=device, non_blocking=non_blocking))

################################

def get_data_loaders(args):
    #TODO make Dali Loaders compatible with the pytorch ones?
    
    traindir = args.data[0]
    valdir= args.data[1]
    
    if(args.arch == "inception_v3"):
        crop_size = 299
        val_size = 320 # I chose this value arbitrarily, we can adjust.
    else:
        crop_size = 224
        val_size = 256
    
    pipe = HybridTrainPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, 
                            data_dir=traindir, crop=crop_size, dali_cpu=False)
    pipe.build()
    train_loader = DALILoader(pipe, ["data", "label"], auto_reset=True, stop_at_epoch=True) 
    #size=int(pipe.epoch_size("Reader") / args.world_size))

    pipe = HybridValPipe(batch_size=args.batch_size, num_threads=args.workers, device_id=args.local_rank, 
                            data_dir=valdir, crop=crop_size, size=val_size)
    pipe.build()
    val_loader = DALILoader(pipe, ["data", "label"], auto_reset=True, stop_at_epoch=True) 
    #size=int(pipe.epoch_size("Reader") / args.world_size))
    
    return train_loader, val_loader


def run(args):
    
    # distributed training variable
    args.world_size = 1
    
    if args.fp16:
        assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."
        
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
            
    if args.arch == 'inception_v3':
        model.aux_logits=False

    device = 'cuda'
    
    ### data loaders
    train_loader, val_loader = get_data_loaders(args)
    
    ### optimizers
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

    # will need my own train loop function to make the fp16 from apex work
    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer,
                                      opt_level=args.opt_level,
                                      keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                                      loss_scale=args.loss_scale
                                      )

    
    if args.wandb_logging:    
        wandb.watch(model)
    
    loss_fn = F.cross_entropy
    accuracy = Accuracy()
    
    scheduler = OneCycleLR(optimizer, num_steps=args.epochs, lr_range=(args.lr/10, args.lr))

    
    # dali loaders need extra class?
    trainer = create_supervised_trainer(model, optimizer, loss_fn,
                                        device=device, non_blocking=False,
                                        prepare_batch=prepare_dali_batch)

    evaluator = create_supervised_evaluator(model,
                                            metrics={'cross_entropy': Loss(loss_fn),
                                                    'accuracy': accuracy},
                                            device=device, non_blocking=False ,
                                            prepare_batch=prepare_dali_batch)

    def score_function(engine):
        accur = engine.state.metrics['accuracy']
        return accur
    
    handler = EarlyStopping(patience=10, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    evaluator.add_event_handler(Events.COMPLETED, handler)

    desc = "ITERATION - loss: {:.2f}"
    pbar = tqdm(
        initial=0, leave=False, total=len(train_loader),
        desc=desc.format(0)
    )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % args.log_interval == 0:
            pbar.desc = desc.format(engine.state.output)
            pbar.update(args.log_interval)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        scheduler.step()
        pbar.refresh()
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['cross_entropy']
        
        # Dali resets
        #train_loader.reset()
        #val_loader.reset()
        
        tqdm.write(
            "Training Results - Epoch: {} LR: {:.2f} Avg loss: {:.2f} Avg accuracy: {:.2f}, Avg Top-5 accuracy: {:.2f}".format(
                    engine.state.epoch, scheduler.get_lr(), avg_loss, avg_accuracy, avg_top5_accuracy)
                
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['cross_entropy']
        avg_top5_accuracy = metrics['top_5_accuracy']

        tqdm.write(
            "Validation Results - Epoch: {} Avg loss: {:.2f} Avg accuracy: {:.2f}, Avg Top-5 accuracy: {:.2f}".format(
                    engine.state.epoch, avg_loss, avg_accuracy, avg_top5_accuracy)
        )

        pbar.n = pbar.last_print_n = 0

    trainer.run(train_loader, max_epochs=args.epochs)
    pbar.close()
    

if __name__ == "__main__":
    
    
    
    parser = ArgumentParser(description="Classification Model Training")
    
    parser.add_argument('data', metavar='DIR', nargs='*',
                    help='paths to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH',
                        choices=valid_models,
                        help='model architecture: | {0} (default: resnet18)'.format(valid_models))
    parser.add_argument('--opt', metavar='OPT', default='sgd',
                        choices=['sgd', 'adam', 'adamw', 'radam'],
                        help='optimiser function')
    parser.add_argument('--num-classes', '-nc', metavar='N', default=10, type=int,
                        help='num classes for classification task (default 10)')
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

    parser.add_argument('--log-interval', type=int, default=20,
                        help='how many steps to take before logging (default:20)')

    parser.add_argument('--wandb-logging', action='store_true')
    
    # keep true unless we vary image sizes
    cudnn.benchmark = True

    args = parser.parse_args()

    if args.wandb_logging:
        
        wandb.init(project="image_classification_opt")
        wandb.config.update(args)
    
    # make apex optional - we aren't using distributed
    if args.fp16: #or args.distributed:
        try:
            #from apex.parallel import DistributedDataParallel as DDP
            from apex.fp16_utils import *
            from apex import amp, optimizers
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this script.")
            
    run(args)