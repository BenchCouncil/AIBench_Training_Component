#!/bin/bash -x

export PYTHONPATH=${PWD}/src

python src/train_tripletloss.py \
    --logs_base_dir ./logs/facenet/ \
    --models_base_dir ./models/VGGFace2-model/ \
    --data_dir ./datasets/casia-webface/casia_maxpy_mtcnnpy_182/ \
    --image_size 160 \
    --model_def models.inception_resnet_v1 \
    --lfw_dir ./datasets/lfw/lfw_mtcnnpy_160/ \
    --optimizer RMSPROP \
    --learning_rate 0.01 \
    --weight_decay 1e-4 \
    --max_nrof_epochs 500

