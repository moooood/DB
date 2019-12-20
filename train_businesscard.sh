#!/bin/bash

#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 nohup python train.py experiments/seg_detector/businesscard_resnet18_conv2d.yaml\
    --resume /data/hongguan.liu/DB/models_on_mlt2017/model/model_epoch_107_minibatch_48000\
    --num_gpus 2\
    >train_businesscard.log 2>&1 &
