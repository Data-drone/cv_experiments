# A train script using just pytroch and torchvision
# test to get training loop right

### TODO
# add checkpoint saving
# try again on FP16
# validation loop
# model watching for wandb
# argparse?
# different models

import torch
import torchvision
import torchvision.models.detection as models
import models.detection as local_models
import argparse

from data_prep.preproc_coco_detect import CocoDetection, CocoDetectProcessor, coco_remove_images_without_annotations
from data_prep.preproc_coco_detect import Compose, RandomHorizontalFlip, ToTensor
from misc_utils.detection_logger import Logger
import os

import wandb

wandb.init(project="object_detection")

### fp16 to save space
try:
    #from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except ImportError:
    raise ImportError("Please install apex from https://www.github.com/nvidia/apex to run this script.")
assert torch.backends.cudnn.enabled, "fp16 mode requires cudnn backend to be enabled."


# this was in the reference code base but need to unpack why?
# the data has to be converted back to list for the detector later anyway?
def collate_fn(batch):
    return tuple(zip(*batch))

def batch_loop(model, optimizer, data_loader, device, epoch, fp16):
    # based on the train_one_epoch detection engine reference script
    model.train()

    if fp16:
        # fp16 fix? - https://github.com/NVIDIA/apex/issues/122
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval().half()

        model.apply(fix_bn)

    metric_logger = Logger()
    header = 'Epoch: [{}]'.format(epoch)

    i = 0
    for images, targets in metric_logger.log(data_loader, header):

        images_l = list(image.to(device) for image in images)
        target_l = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images_l, target_l)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()

        #losses.backward()
        if fp16:
            with amp.scale_loss(losses, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            losses.backward()

        optimizer.step()

        # converting tensors to numbers
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))

        results_dict = loss_dict
        results_dict['epoch'] = epoch
        results_dict['batch'] = i

        wandb.log(results_dict)

        i += 1


def eval_loop(model, optimizer, data_loader, device, epoch, fp16):

    model.eval()
    metric_logger = Logger()
    header = 'Epoch: [{}]'.format(epoch)

    i = 0
    for images, targets in metric_logger.log(data_loader, header):

        images_l = list(image.to(device) for image in images)
        target_l = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(images_l)
        
        #losses = sum(loss for loss in loss_dict.values())

        # converting tensors to numbers
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))

        results_dict = loss_dict
        results_dict['epoch'] = epoch
        results_dict['batch'] = i

        wandb.log(results_dict)

        i += 1


def train(model, optimizer, data_loader, test_loader, device, fp16):

    for epoch in range(10):

        # train one epoch 
        batch_loop(model, optimizer, data_loader, device, epoch, fp16)
        # validate one epoch
        eval_loop(model, optimizer, data_loader, device, epoch, fp16)


def main(args):

    device = torch.device(args.device)
    log_interval = 20

    train_transforms = Compose([CocoDetectProcessor(), ToTensor(), RandomHorizontalFlip(0.5)])
    val_transforms = Compose([CocoDetectProcessor(), ToTensor()])

    ### Coco DataSet Processors
    train_set = CocoDetection(os.path.join(args.data, 'train2017'),
                            os.path.join(args.data, 'annotations', 'instances_train2017.json'), 
                            train_transforms)
    val_set = CocoDetection(os.path.join(args.data, 'val2017'), 
                        os.path.join(args.data, 'annotations', 'instances_val2017.json'), 
                        val_transforms)

    train_set = coco_remove_images_without_annotations(train_set)

    # Coco Dataset Samplers
    train_sampler = torch.utils.data.RandomSampler(train_set)
    test_sampler = torch.utils.data.SequentialSampler(val_set)
    train_batch_sampler = torch.utils.data.BatchSampler(
                train_sampler, args.batch_size, drop_last=True)

    ### pytorch dataloaders
    # cannot increase batch size till we sort the resolutions
    train_loader = torch.utils.data.DataLoader(
            train_set, batch_sampler=train_batch_sampler, num_workers=args.workers,
            collate_fn=collate_fn)

    test_loader = torch.utils.data.DataLoader(
            val_set, batch_size=1,
            sampler=test_sampler, num_workers=args.workers,
            collate_fn=collate_fn)

    # instantiate model
    if args.arch in model_names:
        model = models.__dict__[args.arch](pretrained=False)
    elif args.arch in local_model_names:
        model = local_models.__dict__[args.arch](pretrained=False)

    model.to(device)

    ## declare optimiser
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.SGD(
            params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.fp16:
        model, optimizer = amp.initialize(model, optimizer)

    #wandb.watch(model)

    # trigger train loop

    for epoch in range(10):

        # train one epoch 
        batch_loop(model, optimizer, train_loader, device, epoch, args.fp16)

        # validate one epoch
        #eval_loop(model, optimizer, test_loader, device, epoch, args.fp16)

    #train(model, optimizer, train_loader, test_loader, device, fp_16)



if __name__ == '__main__':

    model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

    local_model_names = sorted(name for name in local_models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(local_models.__dict__[name]))

    valid_models = model_names + local_model_names

    parser = argparse.ArgumentParser(description="PyTorch Detection Model Training")

    parser.add_argument('data', metavar='DIR', default='../external_data/coco',
                        help='paths to dataset')
    parser.add_argument('--arch', '-a', metavar='ARCH',
                    choices=valid_models, default='fasterrcnn_resnet50_fpn',
                    help='model architecture: | {0} (default: fasterrcnn_resnet50_fpn)'.format(valid_models))
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--epochs', '-e', metavar='N', default=10, type=int,
                        help='default num of epochs (default 10)')
    parser.add_argument('-b', '--batch-size', default=3, type=int,
                        metavar='N', help='mini-batch size (default: 3)')
    parser.add_argument('--lr', '--learning-rate', default=0.0025, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('-j', '--workers', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 1)')
    parser.add_argument('--fp16', action='store_true', help='fp 16 or not')                        

    args = parser.parse_args()

    main(args)