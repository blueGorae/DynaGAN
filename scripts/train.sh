#!/bin/bash

PRETRAINED_STYLEGAN_PATH="pretrained_models/ffhq.pt"
IMG_SIZE=1024
TARGET_TRAIN_DATA_DIR_PATH="target_data/raw_data"
DEVICE_NUM=1
ITER=600 # ITER=1800

CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train.py --frozen_gen_ckpt=${PRETRAINED_STYLEGAN_PATH} --size=${IMG_SIZE} --style_img_dir=${TARGET_TRAIN_DATA_DIR_PATH} --iter=${ITER}  --human_face
