#!/bin/bash
# Train DepthBLIP-2 trên SUN3D (indoor RGB-D, similar to NYU)
# SUN3D resolution: 640x480, depth_scale=10000 (0.1mm units), max_depth=10m
export PYTHONUNBUFFERED=1

python ./main.py \
  --train \
  --dataset "SUN3D" \
  --method "second" \
  --auto_bins True \
  --height 480 \
  --width 640 \
  --max_depth 10.0 \
  --depth_scale 10000.0 \
  --data_root_path "./datasets/SUN3D/" \
  --train_file "./datasets/SUN3D/sun3d_train.csv" \
  --val_file   "./datasets/SUN3D/sun3d_val.csv" \
  --test_file  "./datasets/SUN3D/sun3d_test.csv" \
  --train_limit -1 \
  --val_limit -1 \
  --batch_size 2 \
  --workers 2 \
  --epochs 50 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --gpu_num "0" \
  --train_log_save \
  --log_result_dir "./log_results/" \
  --model_save_path "./checkpoints/"
