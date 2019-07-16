#!/bin/bash -x

export PYTHONPATH=${PWD}/src

# face alignment on the LFW dataset
for i in {1..3}; do
    python src/align/align_dataset_mtcnn.py \
    /mnt/sdb/xingw/lfw/lfw/ \
    /mnt/sdb/xingw/lfw/lfw_mtcnnpy_160 \
    --image_size 160 \
    --margin 32 \
    --random_order \
    --gpu_memory_fraction 0.25 \
    &
done

