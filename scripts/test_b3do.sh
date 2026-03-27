#!/bin/bash
# Test DepthBLIP-2 trên B3DO (Berkeley 3-D Object, indoor Kinect)
export PYTHONUNBUFFERED=1

python main.py \
  --gpu_num "0" \
  --method "second" \
  --dataset "B3DO" \
  --auto_bins True \
  --height 480 \
  --width 640 \
  --max_depth 10.0 \
  --depth_scale 10000.0 \
  --data_root_path "./datasets/B3DO/" \
  --test_file "./datasets/B3DO/b3do_test.csv" \
  --class_name "all" \
  --test_log_save \
  --log_result_dir "./log_results" \
  --model_load_path "./checkpoints/second_B3DO_bins_True/train/0/model_epoch_50.pth"
