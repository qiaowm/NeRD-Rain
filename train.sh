#!/bin/bash
python train.py \
    --train_dir '/home/featurize/data/LHP-Rain-RGB-ds/train/' \
    --val_dir '/home/featurize/data/LHP-Rain-RGB-ds/val/' \
    --model_save_dir './checkpoints/LHP-Rain-ds/' \
    --pretrain_weights './checkpoints/model_large_SPA.pth' \
    --num_epochs 50 \
    --batch_size 1 \
    --start_lr 1e-5\
    --end_lr 1e-6 \
    --pretrain \
| tee logs_LHP-Rain.txt