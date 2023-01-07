#!/bin/bash

PRETRAINED_STYLEGAN_PATH="pretrained_models/ffhq.pt"
IMG_SIZE=1024
TARGET_TRAIN_DATA_DIR_PATH="target_data/raw_data"
DEVICE_NUM=1


CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python train.py --frozen_gen_ckpt=${PRETRAINED_STYLEGAN_PATH} --size=${IMG_SIZE} --style_img_dir=${TARGET_TRAIN_DATA_DIR_PATH} --human_face
