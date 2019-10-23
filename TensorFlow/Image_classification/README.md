
# No. DC-AI-C1: Image classification
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This problem domain is to extract different thematic classes within the input data like an image or a text file, which is a supervised learning problem to define a set of target classes and train a model to recognize.

# ResNet for Image Classification

This ResNet-TensorFlow implementation is from tensorflow official models, see https://github.com/tensorflow/models/tree/master/official 

## Environment

Tensorflow 1.12

Cuda 9.0

Cudnn 7.4.2

## Dataset

ImageNet ILSVRC2012 Dataset 

## How to run

### 1.Prepare Dataset
You need to download ImageNet ILSVRC2012 Dataset  from http://www.image-net.org/， then you should convet these raw images to TFRecords by using build_imagenet_data.py script.

### 2.Start running

python imagenet_main.py --data_dir=/path/to/imagenet

Both the training dataset and the validation dataset are in the same directory. 
The model will begin training and will automatically evaluate itself on the validation data roughly once per epoch.

Some options:

--model_dir: to choose where to store the model 

--resnet_size: to choose the model size (options include ResNet-18 through ResNet-200)

--num-gpus: to choose computing device 

            0: Use OneDeviceStrategy and train on CPU
            
            1: Use OneDeviceStrategy and train on GPU.
            
            2+: Use MirroredStrategy (data parallelism) to distribute a batch between devices.

Full list of options, see resnet_run_loop.py


## Refference
[1] [Deep Residual Learning for Image Recognition](https://arxiv.org/pdf/1512.03385.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Dec 2015.

[2] [Identity Mappings in Deep Residual Networks](https://arxiv.org/pdf/1603.05027.pdf) by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun, Jul 2016.
