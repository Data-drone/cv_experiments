import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import models as local_models
from models.layer_utils import Swish, Mish
import numpy as np

import optimisers as local_optimisers

import pytorch_lightning as pl
from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule

from torch.optim.lr_scheduler import CyclicLR
from learning_rate_schedulers.onecyclelr import OneCycleLR

import logging
lightning_console_log = logging.getLogger("lightning")

## lightning wrapper around cvision
class LightningModel(LightningModule):

    def __init__(self, hparams):
        super().__init__()

        model_names = sorted(name for name in models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(models.__dict__[name]))

        local_model_names = sorted(name for name in local_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(local_models.__dict__[name]))

        valid_models = model_names + local_model_names

        self.hparams = vars(hparams) if type(hparams) is not dict else hparams

        if self.hparams['act_func'] == 'swish':
            self.act_funct = Swish()
        elif self.hparams['act_func'] == 'mish':
            self.act_funct = Mish()
        elif self.hparams['act_func'] == 'relu':
            self.act_funct = nn.ReLU(inplace=True)

        # initiate model
        print("=> creating new model '{}'".format(self.hparams['model']))
        if self.hparams['model'] in model_names:
            cv_model = models.__dict__[self.hparams['model']](pretrained=False,
                                                num_classes=self.hparams['num_classes'])
        elif self.hparams['model'] in local_model_names:
            cv_model = local_models.__dict__[self.hparams['model']](pretrained=False,
                                                        activation=self.act_funct,
                                                    num_classes=self.hparams['num_classes'])

        if self.hparams['model'] == 'inception_v3':
            cv_model.aux_logits=False

        self.model = cv_model
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()

        self.tr_accuracy = pl.metrics.Accuracy()
        self.vl_accuracy = pl.metrics.Accuracy()
        self.test_accuracy = pl.metrics.Accuracy()

        # for tensorboard graph logger
        #self.example_input_array = torch.rand((self.input_dim))
        self.example_input_array = torch.rand([1,3,250,250])
        #self.register_buffer("example_input_array", torch.rand([1,3,250,250]))

    def forward(self, x):

        return self.model(x)

    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop with the data from the training dataloader
        passed in as `batch`.
        """
        # forward pass
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y) # this is the criterion
        acc1, acc5 = self.__accuracy(y_hat, y, topk=(1,5))
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_acc', self.tr_accuracy(y_hat, y), on_step=True, logger=True, sync_dist=True)
        self.log('train_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log('train_acc5', acc5, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y) # this is the criterion
        acc1, acc5 = self.__accuracy(y_hat, y, topk=(1, 5))        
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('val_acc', self.vl_accuracy(y_hat, y), on_step=True, logger=True, sync_dist=True)
        self.log('val_acc1', acc1, on_step=True, prog_bar=True, on_epoch=True)
        self.log('val_acc5', acc5, on_step=True, on_epoch=True)

    # from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/imagenet.py
    @staticmethod
    def __accuracy(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def training_epoch_end(self, outs):
        if self.current_epoch==0:
            self.logger[0].experiment.add_graph(self, torch.rand([1,3,250,250]).cuda())
        self.log('train_acc_epoch', self.tr_accuracy.compute(), logger=True, sync_dist=True)

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.vl_accuracy.compute(), logger=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y) # this is the criterion
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('test_acc', self.test_accuracy(y_hat, y), on_step=True, logger=True, sync_dist=True)

    def test_epoch_end(self, outputs):
        self.log('test_acc_epoch', self.test_accuracy.compute(), logger=True, sync_dist=True)

    def on_after_backward(self):
        # adding value histograms for tensorboard
        if self.trainer.global_step % 25 == 0 and self.current_epoch >= 1:  # don't make the tf file huge
            params = self.state_dict()
            for k, v in params.items():
                lightning_console_log.info( "key: {0}, values type: {1}".format(k, v.grad))
                # added to indexing as we have fed in the loggers as a list
                self.logger[0].experiment.add_histogram(
                    # v.grad?
                    tag=k, values=v, global_step=self.trainer.global_step
                )

    #
    # TRAINING SETUP SECTIONS
    #

    def configure_optimizers(self):

        # optimizer
        if self.hparams['opt'] == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), self.hparams['lr'],
                                    momentum=self.hparams['momentum'],
                                    weight_decay=self.hparams['weight_decay'])

        if self.hparams['opt'] == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), self.hparams['lr'],
                                        weight_decay=self.hparams['weight_decay'])

        if self.hparams['opt'] == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), self.hparams['lr'],
                                        weight_decay=self.hparams['weight_decay'])
            
        if self.hparams['opt'] == 'radam':
            optimizer = local_optimisers.RAdam(self.model.parameters(), self.hparams['lr'],
                                            betas=(0.9, 0.999), eps=1e-8,
                                            weight_decay=self.hparams['weight_decay'])

        if self.hparams['opt'] == 'ranger':
            optimizer = local_optimisers.Ranger(self.model.parameters(), self.hparams['lr'],
                                                weight_decay=self.hparams['weight_decay'])

        # scheduler
        #scheduler = CyclicLR(optimizer, 0.00001, self.hparams['lr'])
        scheduler = OneCycleLR(optimizer, num_steps=int(self.hparams['epochs']/2), 
                            lr_range=(self.hparams['lr']/10, self.hparams['lr']))

        schedule = {'scheduler': scheduler, 'interval': 'epoch'}

        return [optimizer], [schedule]

    @staticmethod
    def add_model_specific_args(parent_parser):  # pragma: no-cover
    
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--batch_size', default=64, type=int)
        parser.add_argument('--nworkers', default=4, type=int)

        parser.add_argument('--act_func',
                            type=str,
                            default='relu',
                            choices = ['relu', 'swish', 'mish'],
                            help='enter a valid model name (currently only works for certain models)')

        parser.add_argument('--model',
                            type=str,
                            default='resnet18',
                            help='enter a valid model name')

        parser.add_argument('--opt',
                            type=str,
                            default='sgd',
                            choices=['sgd', 'adam', 'adamw', 'radam', 'ranger'])

        parser.add_argument('--num-classes',
                            type=int,
                            default=100,
                            help='number of classes in classification train')

        return parser