#!/bin/bash

#!/bin/bash

CUDA_VISIBLE_DEVICES=0 nohup python train.py experiments/seg_detector/contract_resnet18_deform_thre.yaml\
    --resume /data/hongguan.liu/DB_models/td500_resnet18\
    --num_gpus 1\
    >train_contract.log 2>&1 &