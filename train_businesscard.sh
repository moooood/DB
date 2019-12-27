#!/bin/bash

#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python train.py experiments/seg_detector/businesscard_resnet18_conv2d.yaml\
    --resume /data/hongguan.liu/DB/models_on_mlt2017/model/model_epoch_37_minibatch_21000\
    --num_gpus 1\
    >train_businesscard.log 2>&1 &
