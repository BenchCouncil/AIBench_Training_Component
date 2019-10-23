# No. DC-AI-C5: Image-to-Image
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This problem domain is to convert an image from one representation of an image to another representation.

# CycleGAN in TensorFlow

The code is provided by Harry Yang (https://github.com/leehomyc/cyclegan-1).


## Getting Started
### Prepare dataset
* You can either download one of the defaults CycleGAN datasets or use your own dataset. 
	* Download a CycleGAN dataset (e.g. horse2zebra):
	```bash
	bash ./download_datasets.sh cityscapes
	```
	* Use your own dataset: put images from each domain at folder_a and folder_b respectively. 

* Create the csv file as input to the data loader. 
	* Edit the cyclegan_datasets.py file. For example, if you have a face2ramen_train dataset which contains 800 face images and 1000 ramen images both in PNG format, you can just edit the cyclegan_datasets.py as following:
	```python
	DATASET_TO_SIZES = {
    'face2ramen_train': 1000
	}

	PATH_TO_CSV = {
    'face2ramen_train': './CycleGAN/input/face2ramen/face2ramen_train.csv'
	}

	DATASET_TO_IMAGETYPE = {
    'face2ramen_train': '.png'
	}

	``` 
	* Run create_cyclegan_dataset.py:
	```bash
	python3 create_cyclegan_dataset.py --image_path_a=./input/cityscapes/trainA --image_path_b=./input/cityscapes/trainB --dataset_name="cityscapes_train" --do_shuffle=0
	```

### Training
* Create the configuration file. The configuration file contains basic information for training/testing. An example of the configuration file could be fond at configs/exp_01.json. 

* Start training:
```bash
python3 main.py \
    --to_train=1 \
    --log_dir=./output/cyclegan/exp_01 \
    --config_filename=./configs/exp_01.json
```
* Check the intermediate results.
	* Tensorboard
	```bash
	tensorboard --port=6006 --logdir=./output/cyclegan/exp_01/#timestamp# 
	```
	* Check the html visualization at ./output/cyclegan/exp_01/#timestamp#/epoch_#id#.html.  

### Restoring from the previous checkpoint.
```bash
python3 main.py \
    --to_train=2 \
    --log_dir=./output/cyclegan/exp_01 \
    --config_filename=./configs/exp_01.json \
    --checkpoint_dir=./output/cyclegan/exp_01/#timestamp#
```
### Testing
* Create the testing dataset.
	* Edit the cyclegan_datasets.py file the same way as training.
	* Create the csv file as the input to the data loader. 
	```bash
	python3 create_cyclegan_dataset.py --image_path_a=./input/cityscapes/testA --image_path_b=./input/cityscapes/testB --dataset_name="cityscapes_test" --do_shuffle=0
	```
* Run testing.
```bash
python3 main.py \
    --to_train=0 \
    --log_dir=./output/cyclegan/exp_01 \
    --config_filename=./configs/exp_01_test.json \
    --checkpoint_dir=./output/cyclegan/exp_01/#old_timestamp# 
```
The result is saved in CycleGAN_TensorFlow/output/cyclegan/exp_01/#new_timestamp#.





