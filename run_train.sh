#!/bin/bash

#python3 train_classification.py -a tinydarknetv3 -nc 10 -e 300 --lr 0.02 -b 128 --fp16 "../cv_data/cifar10/train" "../cv_data/cifar10/test"
#python3 train_classification.py -a tinydarknetv3 -nc 100 -e 300 --lr 0.02 -b 128 --fp16 "../cv_data/cifar100/train" "../cv_data/cifar100/test"

#python3 train_classification.py -a tinydarknetv3 --opt adam -nc 100 -e 150 --lr 0.001 -b 128 --fp16 "../cv_data/cifar100/train" "../cv_data/cifar100/test"

#python train_classification.py -a tinydarknetv3 -nc 10 -e 300 --lr 0.02 -b 128 --fp16 "../cv_data/cifar10/train" "../cv_data/cifar10/test"
#python3 train_classification.py -a tinydarknetv3 -nc 10 -e 300 --lr 0.02 -b 128  "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#### optimisation testing for resnet

#python3 train_classification.py -a resnet18 -nc 10 -e 25 --lr 0.02 -b 256 --fp16 "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a resnet18 -nc 10 -e 25 --lr 0.9 -b 256 --fp16 "../cv_data/cifar10/train" "../cv_data/cifar10/test"
python3 train_classification.py -a resnet18 --opt 'adam' -nc 10 -e 25  --lr 0.02 -b 128 --fp16 "../cv_data/cifar10/train" "../cv_data/cifar10/test"
