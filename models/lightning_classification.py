import os
from argparse import ArgumentParser
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pytorch_lightning import _logger as log
from pytorch_lightning.core import LightningModule
## lightning wrapper around cvision
# Need to test


class LightningModel(LightningModule):

    def __init__(self, hparams, model, optimizer, scheduler, criterion):
        super().__init__()

        # set from hparams?
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

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

        return [self.optimizer], [self.scheduler]

    
    def prepare_data(self, train, test):
        # TODO fix this one
        self.train_data = train
        self.test_data = test

    def train_dataloader(self):
        log.info('Training data loader called.')
        return DataLoader(self.train_data, batch_size=self.hparams.batch_size, num_workers=4)

    def val_dataloader(self):
        log.info('Validation data loader called.')
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, num_workers=4)

    def test_dataloader(self):
        log.info('Test data loader called.')
        return DataLoader(self.test_data, batch_size=self.hparams.batch_size, num_workers=4)

    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no-cover
    
        parser = ArgumentParser(parents=[parent_parser])

        parser.add_argument('--epochs', default=20, type=int)
        parser.add_argument('--batch_size', default=64, type=int)