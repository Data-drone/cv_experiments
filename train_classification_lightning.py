"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import torch

import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only

from models.lightning_classification import LightningModel
from data_pipeline.lightning_dali_loaders import PLDaliPipe
from data_pipeline.basic_lightning_dataloader import BasicPipe

#from lr_schedulers.onecyclelr import OneCycleLR

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

import logging

train_logger = logging.getLogger(__name__ + '.mainLoop')


def choose_dataset(dataset_flag):

    mean_list = {
        'cifar10': (0.4914, 0.4822, 0.4465),
        'cifar100': (0.5071, 0.4867, 0.4408),
        'imagenet': (0.485, 0.456, 0.406)
    }

    std_list = {
        'cifar10': (0.2023, 0.1994, 0.2010),
        'cifar100': (0.2675, 0.2565, 0.2761),
        'imagenet': (0.229, 0.224, 0.225)
    }

    train_list = {
        'cifar10': '../cv_data/cifar10/train',
        'cifar100': '../cv_data/cifar100/train',
        'imagenet': '../external_data/ImageNet/ILSVRC2012_img_train'
    }

    val_list = {
        'cifar10': '../cv_data/cifar10/test',
        'cifar100': '../cv_data/cifar100/test',
        'imagenet': '../external_data/ImageNet/ILSVRC2012_img_val'
    }

    num_classes = {
        'cifar10': 10,
        'cifar100': 100,
        'imagenet': 1000
    }

    mean = mean_list[dataset_flag]
    std = std_list[dataset_flag]
    traindir = train_list[dataset_flag]
    valdir = val_list[dataset_flag]
    num_c = num_classes[dataset_flag] 

    return mean, std, traindir, valdir, num_c


def main(hparams, logger):
    """
    Main training routine specific for this project
    :param hparams:
    """

    # ------------------------
    # Move data loaders out so that the lightning model can be generic
    # ------------------------

    # cifar10 cifar100 imagenet
    mean, std, traindir, valdir, num_classes = choose_dataset('cifar100')
    hparams.num_classes = num_classes

    train_logger.info('Training Directory: {0}'.format(traindir) )

    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------

    model = LightningModel(hparams)

    # -------
    # EARLY STOPPING
    # -------

    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_acc',
        min_delta=0.00,
        patience=15,
        verbose=True,
        mode='max'
    )

    name = '{0}_{1}_cifar100-'.format(hparams.model, hparams.opt)

    save_checkpint_callback = ModelCheckpoint(
        monitor='val_loss',
        filepath = 'saved_model/' + name + '{epoch}-{val_loss:.2f}-{val_acc:.2f}',
        save_top_k = 2,
        mode='min',
        verbose=False
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')

    
    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_16bit else 32,
        min_epochs=50,
        accumulate_grad_batches = 1,
        checkpoint_callback = save_checkpint_callback,
        callbacks=[early_stop_callback, lr_monitor],
        logger=[logger] # , additional_logger
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    #dali_pipe = PLDaliPipe(hparams, traindir, valdir, [*mean], [*std])
    #trainer.fit(model, dali_pipe)
    
    normal_pipe = BasicPipe(hparams, traindir, valdir, mean, std)
    trainer.fit(model, normal_pipe)

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    #root_dir = os.path.dirname(os.path.realpath(__file__))
    #wandb_logger = WandbLogger(project='lightning_test')
    tb_logger = TensorBoardLogger("tb_logs", name="cv_exp")

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
        default='ddp',
        help='supports three options dp, ddp, ddp2'
    )
    parent_parser.add_argument(
        '--use_16bit',
        dest='use_16bit',
        action='store_true',
        help='if true uses 16 bit precision'
    )

    # should I have this for saving configs and models?
    #parent_parser.add_argument('--runname', 'runtest', type=str)


    parent_parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
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
    result = main(hyperparams, tb_logger)
