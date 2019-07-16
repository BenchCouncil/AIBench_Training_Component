#!/bin/bash -x

export PYTHONPATH=${PWD}/src

# face alignment on the CASIA-Webface dataset
for i in {1..3}; do
    python src/align/align_dataset_mtcnn.py \
    /mnt/sdb/xingw/casia-webface/CASIA-WebFace/ \
    /mnt/sdb/xingw/casia-webface/casia_maxpy_mtcnnpy_182 \
    --image_size 182 \
    --margin 44 \
    --random_order \
    --gpu_memory_fraction 0.25 \
    &
done

