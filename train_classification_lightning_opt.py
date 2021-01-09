"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser
from pytorch_lightning.loggers import TensorBoardLogger

import numpy as np
import torch

import pytorch_lightning as pl

from models.lightning_classification import LightningModel
from pytorch_lightning.utilities import rank_zero_only

from data_pipeline.basic_lightning_dataloader import BasicPipe
from train_classification_lightning import choose_dataset

import ray
import shutil
import tempfile
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.cloud_io import load as pl_load
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, \
    TuneReportCheckpointCallback

#from lr_schedulers.onecyclelr import OneCycleLR

SEED = 2334
torch.manual_seed(SEED)
np.random.seed(SEED)

import logging
train_logger = logging.getLogger(__name__ + '.mainLoop')

# ray 1.1.0
# CLI Command: ray start --head --dashboard-host 0.0.0.0 --dashboard-port 8787
ray.init(address='auto')
#, num_cpus=8, num_gpus=2, dashboard_host='0.0.0.0', dashboard_port=8787)

## TODO
## need to reorganise the hparams
## need to rethink
## self.hparams.num_classes is how we access a argparse variable
## self.hparams['num_classes'] is how we access the dict object from ray
## so we would need to move everything to receive dicts...


def tune_main(hparams, num_epochs=15, num_gpus=0):

    print(hparams)

    mean, std, traindir, valdir, num_classes = choose_dataset('cifar10')
    traindir = '/home/jovyan/work/cv_data/cifar10/train'
    valdir = '/home/jovyan/work/cv_data/cifar10/test'
    hparams['num_classes'] = num_classes

    train_logger.info('Training Directory: {0}'.format(traindir) )

    model = LightningModel(hparams)

    trainer = pl.Trainer(
        max_epochs=num_epochs,
        gpus=num_gpus,
        #distributed_backend=hparams.distributed_backend,
        precision=32,
        #early_stop_callback=early_stop_callback,
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        callbacks=[
            TuneReportCallback(
                {
                    "loss": "val_loss_epoch",
                    "accuracy": "val_acc_epoch"
                },
                on="validation_end"
            )
        ])

    normal_pipe = BasicPipe(hparams, traindir, valdir, mean, std)
    trainer.fit(model, normal_pipe)

def tune_cifar_asha(num_samples=10, num_epochs=15, gpus_per_trial=1):
    data_dir = 'test_tune'

    config = {"lr": tune.choice([0.001, 0.01, 0.1]), 
              "act_func": "relu",
              "model": "resnet18",
              "opt": "ranger",
              "epochs": num_epochs,
              "momentum": 0.9,
              "weight_decay": 1e-4,
              "batch_size": 64,
              "nworkers": 4}

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "act_func", "model", "opt", "epochs", "num_classes",
                            "momentum", "weight_decay", "batch_size", "nworkers"],
        metric_columns=["loss", "accuracy", "training_iteration"])

    analysis = tune.run(
        tune.with_parameters(
            tune_main,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial),
        resources_per_trial={
            "cpu": 4,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name="tune_cifar10_asha")

    print("Best hyperparameters found were: ", analysis.best_config)

    #shutil.rmtree(data_dir)

if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments

    #root_dir = os.path.dirname(os.path.realpath(__file__))
    #wandb_logger = WandbLogger(project='lightning_test')
    #logger = TensorBoardLogger("tb_logs", name="cv_exp")

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

    #
    # add tune
    #
    
    # tune hparam configs
    tune_cifar_asha()
    
