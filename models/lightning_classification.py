import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import models as local_models
from models.layer_utils import Swish, Mish

import optimisers as local_optimisers

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule

from torch.optim.lr_scheduler import CyclicLR

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
            cv_model = models.__dict__[self.hparams.model](pretrained=False)
        elif self.hparams.model in local_model_names:
            cv_model = local_models.__dict__[self.hparams.model](pretrained=False,
                                                        activation=self.act_funct)

        if self.hparams.model == 'inception_v3':
            cv_model.aux_logits=False

        self.model = cv_model
        self.criterion = nn.CrossEntropyLoss()


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
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

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
        return {'val_loss': val_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = self.criterion(y_hat, y) # this is the criterion
        labels_hat = torch.argmax(y_hat, dim=1)
        n_correct_pred = torch.sum(y == labels_hat).item()
        return {'test_loss': test_loss, "n_correct_pred": n_correct_pred, "n_pred": len(x)}

    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'val_loss': avg_loss, 'val_acc': val_acc}
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc = sum([x['n_correct_pred'] for x in outputs]) / sum(x['n_pred'] for x in outputs)
        tensorboard_logs = {'test_loss': avg_loss, 'test_acc': test_acc}
        return {'test_loss': avg_loss, 'log': tensorboard_logs}

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
        scheduler = CyclicLR(optimizer, 0.00001, self.hparams.lr)


        return [optimizer], [scheduler]

    
    def prepare_data(self):
        # TODO fix this one
        transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.train_data = torchvision.datasets.CIFAR10(root='../cv_data', train=True,
                                        download=True, transform=transform)
        self.test_data = torchvision.datasets.CIFAR10(root='../cv_data', train=True,
                                        download=True, transform=transform)

    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, num_workers=self.hparams.nworkers)

    def val_dataloader(self):
        log.info('Validation data loader called.')
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, num_workers=self.hparams.nworkers)

    def test_dataloader(self):
        log.info('Test data loader called.')
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, num_workers=self.hparams.nworkers)

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
                            help='enter a valid model name')

        parser.add_argument('--model',
                            type=str,
                            default='resnet18',
                            help='enter a valid model name')

        parser.add_argument('--opt',
                            type=str,
                            default='sgd',
                            choices=['sgd', 'adam', 'adamw', 'radam', 'ranger'])


        return parser