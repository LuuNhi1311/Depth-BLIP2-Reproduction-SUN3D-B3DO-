#!/bin/bash
# Train DepthBLIP-2 trên B3DO (Berkeley 3-D Object, indoor Kinect, 640x480)
# depth_scale=10000 (0.1mm → m, RegisteredDepthData_abs.png format)
export PYTHONUNBUFFERED=1

python ./main.py \
  --train \
  --dataset "B3DO" \
  --method "second" \
  --auto_bins True \
  --height 480 \
  --width 640 \
  --max_depth 10.0 \
  --depth_scale 10000.0 \
  --data_root_path "./datasets/B3DO/" \
  --train_file "./datasets/B3DO/b3do_train.csv" \
  --val_file   "./datasets/B3DO/b3do_val.csv" \
  --test_file  "./datasets/B3DO/b3do_test.csv" \
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
