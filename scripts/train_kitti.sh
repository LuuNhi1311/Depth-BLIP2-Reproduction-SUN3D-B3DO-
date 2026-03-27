#!/bin/bash
# Train DepthBLIP-2 trên KITTI (Eigen split) - tập nhỏ 16 ảnh
# NYU=(416, 544), KITTI=(352, 1216)
export PYTHONUNBUFFERED=1

python ./main.py \
  --train \
  --dataset "KITTI" \
  --method "second" \
  --auto_bins True \
  --height 352 \
  --width 1216 \
  --max_depth 80.0 \
  --data_root_path "./datasets/KITTI/" \
  --train_file "./datasets/KITTI/eigen_train_files_with_gt_dense.txt" \
  --val_file   "./datasets/KITTI/eigen_val_files_with_gt_dense.txt" \
  --test_file  "./datasets/KITTI/eigen_test_files_with_gt_dense.txt" \
  --train_limit 16 \
  --val_limit 8 \
  --batch_size 2 \
  --workers 2 \
  --epochs 50 \
  --lr 0.001 \
  --weight_decay 0.01 \
  --gpu_num "0" \
  --train_log_save \
  --log_result_dir "./log_results/" \
  --model_save_path "./checkpoints/"
