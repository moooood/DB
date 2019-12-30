#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1 nohup python -u train.py experiments/seg_detector/mlt2017_resnet18_conv2d.yaml\
    --num_gpus 2\
    >train_mlt2017.log 2>&1 &
