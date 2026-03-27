#!/bin/bash
# Test DepthBLIP-2 trên SUN3D
export PYTHONUNBUFFERED=1

python main.py \
  --gpu_num "0" \
  --method "second" \
  --dataset "SUN3D" \
  --auto_bins True \
  --height 480 \
  --width 640 \
  --max_depth 10.0 \
  --depth_scale 10000.0 \
  --data_root_path "./datasets/SUN3D/" \
  --test_file "./datasets/SUN3D/sun3d_test.csv" \
  --class_name "all" \
  --test_log_save \
  --log_result_dir "./log_results" \
  --model_load_path "./checkpoints/second_SUN3D_bins_True/train/0/model_epoch_50.pth"
