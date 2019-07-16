# 8. Object Detection Faster R-CNN (PyTorch)
## Preparation

### Environment Preparation
```bash
pip install -r requirements.txt
cd lib && python setup.py build develop
```

### Download Datasets and Pretrained Models
To download `Microsoft COCO2014`, please refer to [HERE](https://github.com/XingwXiong/AIBench/tree/master/8-ObjectDetection/caffe#coco-2014)

```bash
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
if [[ ! -f data/coco/images  || ! -h data/coco/images ]]; then
    ln -sv $COCO_PATH data/coco/images
fi
```

## Usages
### Train
```bash
# Note: you can update the parameters in `scripts/train.sh`
# Here, my GPU is Tesla M40 with 24G memory, and I set `BATCH_SIZE=10`
# and `WORKER_NUMBER=10` 

# CUDA_VISIBLE_DEVICES=$GPU_ID python trainval_net.py \
#                    --dataset coco --net res101 \
#                    --bs $BATCH_SIZE --nw $WORKER_NUMBER \
#                    --lr $LEARNING_RATE --lr_decay_step $DECAY_STEP \
#                    --cuda
./scripts/train.sh
```
### Test
Please refer to [HERE](https://github.com/jwyang/faster-rcnn.pytorch#test).

## References
1. [ICCV2015 Tutorial: Training R-CNNs of various velocities](https://www.dropbox.com/s/xtr4yd4i5e0vw8g/iccv15_tutorial_training_rbg.pdf?dl=0)
1. [https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0](https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0)
