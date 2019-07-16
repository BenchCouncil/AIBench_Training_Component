#!/bin/bash -x

export PYTHONPATH=${PWD}/src

python src/validate_on_lfw.py \
    ./datasets/lfw/lfw_mtcnnpy_160 \
    ./models/VGGFace2-model \
    --distance_metric 1 \
    --use_flipped_images \
    --subtract_mean \
    --use_fixed_image_standardization
