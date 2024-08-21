#!/usr/bin/env bash
GPU_DEVICE="0"
length=${#GPU_DEVICE}
n_gpu=$(((length+1)/2))

# (Stage1) Pipeline
CUDA_VISIBLE_DEVICES=$GPU_DEVICE \
accelerate launch --config_file utils/ddp_accel_fp16.yaml \
--num_processes=$n_gpu \
test_closed_models.py \
--batch_size 64 \
--model gpt \
--multiprocess True \