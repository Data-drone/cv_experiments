"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning import Callback
from pytorch_lightning.logging import LightningLoggerBase
from pytorch_lightning.utilities import rank_zero_only

from models.lightning_classification import LightningModel

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

#from lr_schedulers.onecyclelr import OneCycleLR

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)


class DictLogger(LightningLoggerBase):
    """Custom logger to get metrics back"""

    def __init__(self, version):
        super(DictLogger, self).__init__()
        self.metrics = []
        self._version = version

    @rank_zero_only
    def log_metrics(self, metrics, step=None):
        self.metrics.append(metrics)
    
    @property
    def version(self):
        return self._version

    @property
    def experiment(self):
        """Return the experiment object associated with this logger."""

    @property
    def name(self):
        """Return the experiment name."""
        return 'optuna'

    @rank_zero_only
    def log_hyperparams(self, params):
        """
        Record hyperparameters.
        Args:
            params: :class:`~argparse.Namespace` containing the hyperparameters
        """

    
class MetricsCallback(Callback):
    """PyTorch Lightning metric callback."""

    def __init__(self):
        super().__init__()
        self.metrics = []

    def on_validation_end(self, trainer, pl_module):
        self.metrics.append(trainer.callback_metrics)


def main(hparams, logger):
    """
    Main training routine specific for this project
    :param hparams:
    """

    # ------------------------
    # Move data loaders out so that the lightning model can be generic
    # ------------------------

    data_transform = transforms.Compose([
            transforms.Resize((300,300)),
            transforms.CenterCrop((100, 100)),
            transforms.RandomCrop((80, 80)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=(-90, 90)),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_data = ImageFolder(
        root=os.path.join('../cv_data/cifar100', 'train'),
        transform = data_transform
    )

    val_data = ImageFolder(
        root=os.path.join('../cv_data/cifar100', 'test'),
        transform = data_transform
    )

    train_loader = DataLoader(train_data, batch_size=hparams.batch_size, num_workers=hparams.nworkers)
    val_loader = DataLoader(val_data, batch_size=hparams.batch_size, num_workers=hparams.nworkers)

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------

    model = LightningModel(hparams)

    # -------
    # EARLY STOPPING
    # -------

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )

    metrics_callback = MetricsCallback()
    additional_logger = DictLogger('1')
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_16bit else 32,
        #callbacks = [metrics_callback],
        #early_stop_callback=early_stop_callback,
        logger=[logger, additional_logger]
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model, train_dataloader = train_loader,
                        val_dataloaders = val_loader)

    return additional_logger #.metrics[-1]["val_loss"]


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    #root_dir = os.path.dirname(os.path.realpath(__file__))
    #wandb_logger = WandbLogger(project='lightning_test')
    logger = TensorBoardLogger("tb_logs", name="cv_exp")

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
    result = main(hyperparams, logger)

    # doesn't work
    #print(trained_model.test_result())
