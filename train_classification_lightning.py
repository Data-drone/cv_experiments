"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn


import pytorch_lightning as pl
from models.lightning_classification import LightningModel
import torchvision.models as models
import models as local_models
import optimisers as local_optimisers

from lr_schedulers.onecyclelr import OneCycleLR
from torch.optim.lr_scheduler import CyclicLR
from models.layer_utils import Swish, Mish

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

local_model_names = sorted(name for name in local_models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(local_models.__dict__[name]))

valid_models = model_names + local_model_names


def main(hparams):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------

    # activation function
    if hparams.act_func == 'swish':
        act_funct = Swish()
    elif hparams.act_func == 'mish':
        act_funct = Mish()
    elif hparams.act_func == 'relu':
        act_funct = nn.ReLU(inplace=True)


    # initiate model
    print("=> creating new model '{}'".format(hparams.model))
    if hparams.model in model_names:
        cv_model = models.__dict__[hparams.model](pretrained=False)
    elif hparams.model in local_model_names:
        cv_model = local_models.__dict__[hparams.model](pretrained=False,
                                                    activation=act_funct)

    if hparams.model == 'inception_v3':
        model.aux_logits=False

    # optimizer
    if hparams.opt == 'sgd':
        optimizer = torch.optim.SGD(cv_model.parameters(), hparams.lr,
                                momentum=hparams.momentum,
                                weight_decay=hparams.weight_decay)

    if hparams.opt == 'adam':
        optimizer = torch.optim.Adam(cv_model.parameters(), hparams.lr,
                                    weight_decay=hparams.weight_decay)

    if hparams.opt == 'adamw':
        optimizer = torch.optim.AdamW(cv_model.parameters(), hparams.lr,
                                    weight_decay=hparams.weight_decay)
        
    if hparams.opt == 'radam':
        optimizer = local_optimisers.RAdam(cv_model.parameters(), hparams.lr,
                                          betas=(0.9, 0.999), eps=1e-8,
                                           weight_decay=hparams.weight_decay)

    if hparams.opt == 'ranger':
        optimizer = local_optimisers.Ranger(cv_model.parameters(), hparams.lr,
                                            weight_decay=hparams.weight_decay)

    # scheduler
    scheduler = CyclicLR(optimizer, 0.00001, hparams.lr)

    # criterion
    criterion = nn.CrossEntropyLoss()

    # send to lightning
    model = LightningModel(hparams, cv_model, optimizer, scheduler, criterion)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_16bit else 32,
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    #root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # gpu args
    parent_parser.add_argument(
        '--gpus',
        type=int,
        default=2,
        help='how many gpus'
    )
    parent_parser.add_argument(
        '--distributed_backend',
        type=str,
        default='dp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )

    parent_parser.add_argument(
        '--model',
        type=str,
        default='resnet18',
        help='enter a valid model name'
    )

    parent_parser.add_argument(
        '--act_func',
        type=str,
        default='relu',
        choices = ['relu', 'swish', 'mish'],
        help='enter a valid model name'
    )

    parent_parser.add_argument(
        '--opt',
        type=str,
        default='adam',
        choices=['sgd', 'adam', 'adamw', 'radam', 'ranger']
    )

    parent_parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='Initial learning rate.  Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256.  A warmup schedule will also be applied over the first 5 epochs.')
    parent_parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
    parent_parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')


    # each LightningModule defines arguments relevant to it
    parser = LightningModel.add_model_specific_args(parent_parser)
    hyperparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hyperparams)
