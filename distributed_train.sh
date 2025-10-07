#!/bin/bash

NUM_PROC=$1                # 获取第一个参数，作为进程数
# shift                       # 移除第一个参数
torchrun --nproc_per_node=$NUM_PROC /kaggle/working/pytorch-image-models/train.py \
  --config="./kaggle/working/train_config.yaml" \
  --no-prefetcher \
  # --fold=1 \
  "$@"                       # 将其余参数传递给 train.py
