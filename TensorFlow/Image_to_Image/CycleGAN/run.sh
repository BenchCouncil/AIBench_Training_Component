python3 create_cyclegan_dataset.py --image_path_a=./input/cityscapes/trainA --image_path_b=./input/cityscapes/trainB --dataset_name="cityscapes_train" --do_shuffle=0
python3 main.py --to_train=1 --log_dir=./output/cyclegan/exp_01 --config_filename=./configs/exp_01.json
