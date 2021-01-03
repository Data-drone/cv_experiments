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

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule

from torch.optim.lr_scheduler import CyclicLR
from learning_rate_schedulers.onecyclelr import OneCycleLR


## lightning wrapper around cvision
# Need to test


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

        self.hparams = hparams

        if self.hparams.act_func == 'swish':
            self.act_funct = Swish()
        elif self.hparams.act_func == 'mish':
            self.act_funct = Mish()
        elif self.hparams.act_func == 'relu':
            self.act_funct = nn.ReLU(inplace=True)

        # initiate model
        print("=> creating new model '{}'".format(self.hparams.model))
        if self.hparams.model in model_names:
            cv_model = models.__dict__[self.hparams.model](pretrained=False,
                                                num_classes=self.hparams.num_classes)
        elif self.hparams.model in local_model_names:
            cv_model = local_models.__dict__[self.hparams.model](pretrained=False,
                                                        activation=self.act_funct,
                                                    num_classes=self.hparams.num_classes)

        if self.hparams.model == 'inception_v3':
            cv_model.aux_logits=False

        self.model = cv_model
        self.criterion = nn.CrossEntropyLoss()

        self.save_hyperparameters()
        

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

        #print('target = {0}'.format(y.shape))
        #print('predictions = {0}'.format(y_hat.shape))

        loss = self.criterion(y_hat, y) # this is the criterion
        #print(type(loss))
        # pre 1.0
        #return {'loss': loss, 'log': tensorboard_logs}
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop with the data from the validation dataloader
        passed in as `batch`.
        """
        x, y = batch
        y_hat = self(x)
        val_loss = self.criterion(y_hat, y) # this is the criterion
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        
        self.log('val_loss', val_loss, on_step=True, on_epoch=True, sync_dist=True)
        metrics = {"n_correct_pred": n_correct_pred, "n_pred": len(x)}
        self.log_dict(metrics)
        
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y) # this is the criterion
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        self.log('test_loss', test_loss, on_step=True, on_epoch=True, sync_dist=True)
        metrics = {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}
        self.log_dict(metrics)
        
        return {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, validation_step_outputs):
        
        avg_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in validation_step_outputs]) / sum(x['n_pred'] for x in validation_step_outputs)
        #val_acc = sum(corr_pred)/(len(validation_step_outputs) * self.hparams.batch_size) 
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        metrics = {'val_loss': avg_loss, 'val_acc': val_acc, 'log': tensorboard_logs}
        self.log_dict(metrics, logger=True)
        
    # need to restructure for 1.1.2
    #def validation_epoch_end(self, outputs):
    #    """
    #    Called at the end of validation to aggregate outputs.
    #    :param outputs: list of individual outputs of each validation step.
    #    """
    #    #avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    #    #val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
    #    tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
    #    metrics = {'val_loss': avg_loss, 'val_acc': val_acc, 'log': tensorboard_logs}
    #    self.log_dict(metrics)
    #    #return {'val_loss': avg_loss, 'val_acc': val_acc, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        logs = {'test_loss': avg_loss, 'test_acc': test_acc}
        self.log_dict(logs, logger=True)

    #def test_epoch_end(self, outputs):
    #    avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
    #    test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
    #    tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc}
    #    metrics = {'test_loss': avg_loss, 'test_acc': test_acc, 'log': tensorboard_logs}
    #    self.log_dict(metrics)
    #    #return {'test_loss': avg_loss, 'test_acc': test_acc, 'log': tensorboard_logs}


    #
    # TRAINING SETUP SECTIONS
    #

    def configure_optimizers(self):

        # optimizer
        if self.hparams.opt == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), self.hparams.lr,
                                    momentum=self.hparams.momentum,
                                    weight_decay=self.hparams.weight_decay)

        if self.hparams.opt == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), self.hparams.lr,
                                        weight_decay=self.hparams.weight_decay)

        if self.hparams.opt == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), self.hparams.lr,
                                        weight_decay=self.hparams.weight_decay)
            
        if self.hparams.opt == 'radam':
            optimizer = local_optimisers.RAdam(self.model.parameters(), self.hparams.lr,
                                            betas=(0.9, 0.999), eps=1e-8,
                                            weight_decay=self.hparams.weight_decay)

        if self.hparams.opt == 'ranger':
            optimizer = local_optimisers.Ranger(self.model.parameters(), self.hparams.lr,
                                                weight_decay=self.hparams.weight_decay)

        # scheduler
        #scheduler = CyclicLR(optimizer, 0.00001, self.hparams.lr)
        scheduler = OneCycleLR(optimizer, num_steps=int(self.hparams.epochs/2), 
                            lr_range=(self.hparams.lr/10, self.hparams.lr))

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