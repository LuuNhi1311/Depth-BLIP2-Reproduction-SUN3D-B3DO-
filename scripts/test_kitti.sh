#!/bin/bash
# Test DepthBLIP-2 trên KITTI
export PYTHONUNBUFFERED=1

python main.py \
  --gpu_num "0" \
  --method "second" \
  --dataset "KITTI" \
  --auto_bins True \
  --height 352 \
  --width 1216 \
  --max_depth 80.0 \
  --data_root_path "./datasets/KITTI/" \
  --test_file "./datasets/KITTI/eigen_test_files_with_gt_dense.txt" \
  --class_name "all" \
  --test_log_save \
  --log_result_dir "./log_results" \
  --model_load_path "./checkpoints/second_KITTI_bins_True/train/0/model_epoch_50.pth"
