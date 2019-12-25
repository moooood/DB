#!/bin/bash

CUDA_VISIBLE_DEVICES=1 nohup python train.py experiments/seg_detector/mlt2017_resnet18_conv2d.yaml\
    --num_gpus 1\
    >train_mlt17.log 2>&1 &
