#!/bin/bash

python3 train_classification_lightning.py --gpus 2 --distributed_backend ddp --model resnet18 --epochs 50 --nworkers 8

python3 train_classification_lightning.py --gpus 2 --distributed_backend ddp --model vgg16  --epochs 50 --nworkers 8

python3 train_classification_lightning.py --gpus 2 --distributed_backend ddp --model squeezenet1_0 --epochs 50 --nworkers 8

python3 train_classification_lightning.py --gpus 2 --distributed_backend ddp --model movilenet_v2 --epochs 50 --nworkers 8

python3 train_classification_lightning.py --gpus 2 --distributed_backend ddp --model resnet50 --epochs 50 --nworkers 8

python3 train_classification_lightning.py --gpus 2 --distributed_backend ddp --model resnext50_32x4d --epochs 50 --nworkers 8