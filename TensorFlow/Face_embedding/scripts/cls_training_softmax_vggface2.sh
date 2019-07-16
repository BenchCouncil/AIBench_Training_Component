#!/bin/bash -x

export PYTHONPATH=${PWD}/src

python src/train_softmax.py \
    --logs_base_dir ./logs/facenet/ \
    --models_base_dir ./models/VGGFace2-model/ \
    --data_dir ./datasets/vggface2/vggface2_train_182/ \
    --image_size 160 \
    --model_def models.inception_resnet_v1 \
    --lfw_dir ./datasets/lfw/lfw_mtcnnpy_160/ \
    --optimizer ADAM \
    --learning_rate -1 \
    --max_nrof_epochs 500 \
    --batch_size 90 \
    --keep_probability 0.4 \
    --random_flip \
    --use_fixed_image_standardization \
    --learning_rate_schedule_file data/learning_rate_schedule_classifier_vggface2.txt \
    --weight_decay 5e-4 \
    --embedding_size 512 \
    --lfw_distance_metric 1 \
    --lfw_use_flipped_images \
    --lfw_subtract_mean \
    --validation_set_split_ratio 0.01 \
    --validate_every_n_epochs 5

