#!/bin/bash

LATENT_PATH="test_latent.npy"
CKPT_PATH="output/checkpoint/final.pt"

CUDA_VISIBLE_DEVICES=${DEVICE_NUM} python generate.py --ckpt=${CKPT_PATH} --latent_path=${LATENT_PATH}
