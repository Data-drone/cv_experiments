#!/bin/bash

python3 train_classification.py -a tinydarknetv3 -nc 10 --lr 0.02 -b 64 --fp16 "../cv_data/cifar10/train" "../cv_data/cifar10/test"