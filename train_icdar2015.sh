#!/bin/bash

#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup python train.py experiments/seg_detector/icdar2015_resnet18_conv2d.yaml\
    --resume /data/hongguan.liu/DB_models/pre-trained-model-synthtext-resnet18\
    --num_gpus 1\
    >train_icdar2015.log 2>&1 &
