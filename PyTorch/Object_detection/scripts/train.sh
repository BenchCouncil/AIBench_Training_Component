#!/bin/bash

SCRIPTS_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
DIR=$(cd $(dirname ${BASH_SOURCE[0]})/../faster-rcnn.pytorch && pwd)

# go into directory of `faster-rcnn.pytorch`
cd $DIR

# set your own coco dataset path
COCO_PATH=/home/share/coco2014

mkdir -p data data/pretrained_model
set -x
if [[ ! -d data/coco ]]; then
    cd data
    git clone https://github.com/pdollar/coco.git && cd coco/PythonAPI && make -j32 && cd ../../
    cd ../
fi
if [[ ! -f data/coco/annotations || ! -h data/coco/annotations ]]; then
    ln -sv $COCO_PATH/annotations data/coco/annotations
fi
if [[ ! -f data/coco/images || ! -h data/coco/images ]]; then
    ln -sv $COCO_PATH data/coco/images
fi

set +x


# Trainning Configurations
GPU_ID=0
BATCH_SIZE=10
WORKER_NUMBER=10
LEARNING_RATE=0.001
DECAY_STEP=5


# CPU mode
#python trainval_net.py \
#    --dataset coco --net res101 \
#    --bs $BATCH_SIZE --nw $WORKER_NUMBER \
#    --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP

# GPU mode
CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
                    --dataset coco --net res101 \
                    --bs $BATCH_SIZE --nw $WORKER_NUMBER \
                    --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
                    --disp_interval 1 \
                    --cuda
