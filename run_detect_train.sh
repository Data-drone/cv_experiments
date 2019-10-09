#!/bin/bash

#python3 train_detection_basic.py "../external_data/coco"

python3 multiproc.py --nproc_per_node 2 train_detection_basic.py -a 'fasterrcnn_mobilenetv2_fpn' -b 5 --fp16 --opt-level O1 "../external_data/coco"
#python3 train_detection_basic.py --fp16 "../external_data/coco"
