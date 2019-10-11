#!/bin/bash

### imagenet test
#python3 train_classification.py -a resnet18 --opt 'ranger' -nc 1000 -e 10  --lr 0.02 -b 256 --fp16 --opt-level 'O1' --label-smoothing 0.1 "../external_data/ImageNet/ILSVRC2012_img_train" "../external_data/ImageNet/ILSVRC2012_img_val"


#python3 train_classification.py -a tinydarknetv3 -nc 10 -e 300 --lr 0.02 -b 128 --fp16 "../cv_data/cifar10/train" "../cv_data/cifar10/test"
#python3 train_classification.py -a tinydarknetv3 -nc 100 -e 300 --lr 0.02 -b 128 --fp16 "../cv_data/cifar100/train" "../cv_data/cifar100/test"

#python3 train_classification.py -a tinydarknetv3 --opt adam -nc 100 -e 150 --lr 0.001 -b 128 --fp16 "../cv_data/cifar100/train" "../cv_data/cifar100/test"

#python train_classification.py -a tinydarknetv3 -nc 10 -e 300 --lr 0.02 -b 128 --fp16 "../cv_data/cifar10/train" "../cv_data/cifar10/test"
#python3 train_classification.py -a tinydarknetv3 -nc 10 -e 300 --lr 0.02 -b 128  "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#### optimisation testing for resnet

#python3 train_classification.py -a resnet18 -nc 10 -e 25 --lr 0.02 -b 256 --fp16 "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a resnet18 -nc 10 -e 25 --lr 0.9 -b 256 --fp16 "../cv_data/cifar10/train" "../cv_data/cifar10/test"
#python3 train_classification.py -a resnet18 --opt 'adam' -nc 10 -e 25  --lr 0.02 -b 256 --fp16 "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a se_resnet18 --opt 'ranger' -nc 10 -e 10  --lr 0.02 -b 256 --fp16 --opt-level 'O1' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a se_resnet18 --data-backend 'dali-cpu' --opt 'ranger' -nc 1000 -e 10  --lr 0.02 -b 256 --fp16 --opt-level 'O1' "../external_data/ImageNet/ILSVRC2012_img_train" "../external_data/ImageNet/ILSVRC2012_img_val"
#python3 train_classification.py -a wide_resnet18 --data-backend 'dali-cpu' --opt 'ranger' -nc 10 -e 10  --lr 0.02 -b 64 --fp16 --opt-level 'O1' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 multiproc.py --nproc_per_node 2 train_classification.py -a wide_resnet22_2 --data-backend 'dali-gpu' --opt 'ranger' -nc 10 -e 8  --lr 0.02 -b 128 --fp16 --opt-level 'O2' "../cv_data/cifar10/train" "../cv_data/cifar10/test"
python3 multiproc.py --nproc_per_node 2 train_classification.py -a wide_resnet28_10 --data-backend 'dali-gpu' --opt 'sgd' -nc 10 -e 30  --lr 0.1 -b 32 --fp16 --opt-level 'O2' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

# 
#python3 train_classification.py -a resnet18 --data-backend 'dali-cpu' --opt 'ranger' -nc 100 -e 10  --lr 0.02 -b 256 --fp16 --opt-level 'O1' "../cv_data/cifar100/train" "../cv_data/cifar100/test"
#python3 train_classification.py -a resnet18 --data-backend 'dali-cpu' --opt 'ranger' -nc 1000 -e 10  --lr 0.02 -b 256 --fp16 --opt-level 'O1' "../external_data/ImageNet/ILSVRC2012_img_train" "../external_data/ImageNet/ILSVRC2012_img_val"

#python3 train_classification.py -a optresnet18 --opt 'ranger' -nc 10 -e 10  --lr 0.02 -b 256 --fp16 --opt-level 'O1' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a se_resnet18 --opt 'sgd' -nc 100 -e 20  --lr 0.02 -b 4 --fp16 --opt-level 'O2' "../cv_data/cifar100/train" "../cv_data/cifar100/test"


# causes overflow
#python3 train_classification.py -a optresnet18 --opt 'adam' -nc 10 -e 25  --lr 0.9 -b 256 --fp16 --opt-level 'O1' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#
#python3 train_classification.py -a optresnet18 --opt 'sgd' -nc 10 -e 25  --lr 0.1 -b 256 --fp16 --opt-level 'O2' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a optresnet18 --opt 'sgd' -nc 10 -e 25  --lr 0.2 -b 256 --fp16 --opt-level 'O2' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a optresnet18 --opt 'sgd' -nc 10 -e 25  --lr 0.3 -b 256 --fp16 --opt-level 'O2' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a optresnet18 --opt 'sgd' -nc 10 -e 25  --lr 0.4 -b 256 --fp16 --opt-level 'O2' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a optresnet18 --opt 'sgd' -nc 10 -e 25  --lr 0.5 -b 256 --fp16 --opt-level 'O2' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a optresnet18 --opt 'sgd' -nc 100 -e 30  --lr 0.5 -b 256 --fp16 --opt-level 'O2' "../cv_data/cifar100/train" "../cv_data/cifar100/test"

## Trying to find my time for 94% accuracy on val
#python3 train_classification.py -a optresnet18 --opt 'sgd' -nc 10 -e 12  --lr 0.5 -b 256 --fp16 --opt-level 'O2' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a resnet18 --opt 'ranger' -nc 10 -e 10  --lr 0.1 -b 384 --fp16 --opt-level 'O2' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a optresnet18 --opt 'ranger' -nc 10 -e 10  --lr 0.1 -b 384 --fp16 --opt-level 'O2' "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a se_resnet18 --opt 'ranger' -nc 100 -e 60  --lr 0.1 -b 128 --fp16 --opt-level 'O2' "../cv_data/cifar100/train" "../cv_data/cifar100/test"

#python3 train_classification.py -a se_resnet34 --opt 'ranger' -nc 100 -e 60  --lr 0.1 -b 128 --fp16 --opt-level 'O2' "../cv_data/cifar100/train" "../cv_data/cifar100/test"

### ignite testing
#python3 train_classification_ignite.py -a optresnet18 --opt 'radam' -nc 10 -e 3  --lr 0.1 -b 128 --fp16 --opt-level 'O0'  "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification_ignite.py -a optresnet18 --opt 'radam' -nc 10 -e 3  --lr 0.1 -b 128 --fp16 --opt-level 'O1'  "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification_ignite.py -a optresnet18 --opt 'radam' -nc 10 -e 3  --lr 0.1 -b 128 --fp16 --opt-level 'O2'  "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification_ignite.py -a optresnet18 --opt 'radam' -nc 10 -e 3  --lr 0.1 -b 128 --fp16 --opt-level 'O3'  "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#### Label smoothing

# need revisit
#python3 train_classification.py -a attn_resnet18 --opt 'sgd' -nc 10 -e 10  --lr 0.02 -b 4 --fp16 --opt-level 'O2' --label-smoothing 0.5 "../cv_data/cifar10/train" "../cv_data/cifar10/test"

#python3 train_classification.py -a resnet18 --opt 'ranger' -nc 10 -e 10  --lr 0.02 -b 256 --fp16 --opt-level 'O1' --label_smoothing --smoothing 0.1 "../cv_data/cifar10/train" "../cv_data/cifar10/test"


#python3 train_classification.py -a resnet18 --opt 'ranger' -nc 10 -e 10  --lr 0.02 -b 256 --fp16 --opt-level 'O1' --label_smoothing --smoothing 0.1 "../cv_data/cifar10/train" "../cv_data/cifar10/test"


#python3 train_classification.py -a resnet18 --opt 'ranger' -nc 10 -e 10  --lr 0.02 -b 256 --fp16 --opt-level 'O1' --label_smoothing --smoothing 0.1 "../cv_data/cifar10/train" "../cv_data/cifar10/test"


#python3 train_classification.py -a resnet18 --opt 'ranger' -nc 10 -e 10  --lr 0.02 -b 256 --fp16 --opt-level 'O1' --label_smoothing --smoothing 0.1 "../cv_data/cifar10/train" "../cv_data/cifar10/test"

