#!/bin/bash
rm -r ./ckpt/
python train.py \
  --gpus 0 \
  --dset_dir ./datasets \
  --n_workers 4 \
  --ckpt_dir ./ckpt \
  --dset_name moving_mnist \
  --evaluate_every 20 \
  --lr_init 1e-3 \
  --lr_decay 1 \
  --n_iters 200 \
  --batch_size 64 \
  --n_components 2 \
  --stn_scale_prior 2 \
  --ckpt_name 200k
