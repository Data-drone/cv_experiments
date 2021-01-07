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

from ray import tune
import ray
from ray.tune.integration.pytorch_lightning import TuneReportCallback

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
## execute main from a jupyter and see if model.test_result gets us accuracy 

from train_classification_lightning import choose_dataset

def tune_main():
    pass

def main(hparams, logger):
    """
    Main training routine specific for this project
    :param hparams:
    """

    # ------------------------
    # Choose Dataset
    # ------------------------
    mean, std, traindir, valdir, num_classes = choose_dataset('cifar10')
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
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=True,
        mode='min'
    )

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = pl.Trainer(
        max_epochs=hparams.epochs,
        gpus=hparams.gpus,
        distributed_backend=hparams.distributed_backend,
        precision=16 if hparams.use_16bit else 32,
        #early_stop_callback=early_stop_callback,
        logger=logger
    )

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    normal_pipe = BasicPipe(hparams, traindir, valdir, mean, std)
    trainer.fit(model, normal_pipe)



def tune_model(config):

    main(config['hparams'], config['logger'])


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
    callback = TuneReportCallback({
        "loss": "val_loss_epoch",
        "accuracy": "val_acc_epoch"
    }, on="validation_end")

    # tune hparam configs
    tune_configs = {"lr": tune.grid_search([0.001, 0.01, 0.1]), 
                            "hparams": hyperparams,
                            "logger": logger}

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=1,
        reduction_factor=2)

    reporter = CLIReporter(
        parameter_columns=["lr", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    analysis = tune.run(
        tune_model, config=tune_configs, resources_per_trial={'gpu': 1, 'cpu': 4})

    
