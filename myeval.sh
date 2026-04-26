#!/usr/bin/env bash
set -e

# 使用的 GPU
export CUDA_VISIBLE_DEVICES=0

# 模型 checkpoint 路径（按你的实际路径改）
CKPT="./outputs/thyroid_lsnet_t/lsnet_t/2026_02_26_21_51_51/checkpoint_best.pth"

# 多个测试集根目录（ImageFolder 结构，每个下面有两个类别文件夹）
TEST1="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/DDTI_Classification/all_cls"
TEST2="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN3K/test_cls"
TEST3="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/ThyroidXL/test_cls"
TEST4="/mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/TN5K/test_cls"

python eval_thyroid.py \
  --checkpoint "${CKPT}" \
  --test-dirs "${TEST1}" "${TEST2}" "${TEST3}" "${TEST4}"\
  --names "DDTI" "TN3K" "ThyroidXL" "TN5K" \
  --batch-size 16 \
  --device cuda \
  --pos-class-index 1 \
  --log-file logs/thyroid_eval.log