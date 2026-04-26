export CUDA_VISIBLE_DEVICES=1

CUDA_VISIBLE_DEVICES=0 python main.py \
  --model lsnet_t \
  --batch-size 256 \
  --epochs 40 \
  --data-set THYROID \
  --data-path /mnt/wangbd8/workspace/DataSets/ThyroidAgent/train_val_test/Superimposed_multitask/dataset_3_cls \
  --finetune pretrain/lsnet_t.pth \
  --output_dir outputs/thyroid_lsnet_t