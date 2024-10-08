#!/usr/bin/env bash
GPU_DEVICE="0,1,2,3,4,5,6,7"
length=${#GPU_DEVICE}
n_gpu=$(((length+1)/2))

# (Stage1) Pipeline
CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
accelerate launch --config_file utils/ddp_accel_fp16.yaml \
--num_processes=$n_gpu \
test.py \
--batch_size 1 \
--model llava \
