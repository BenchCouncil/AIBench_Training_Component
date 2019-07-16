#!/bin/bash -x

export PYTHONPATH=${PWD}/src

# face alignment on the LFW dataset
for i in {1..3}; do
    python src/align/align_dataset_mtcnn.py \
    ./datasets/vggface2/train/ \
    ./datasets/vggface2/vggface2_train_182 \
    --image_size 182 \
    --margin 44 \
    --random_order \
    --gpu_memory_fraction 0.25 \
    &
done

