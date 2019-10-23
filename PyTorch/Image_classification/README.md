# No. DC-AI-C1: Image classification
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This problem domain is to extract different thematic classes within the input data like an image or a text file, which is a supervised learning problem to define a set of target classes and train a model to recognize.

## Requirements


- Install PyTorch ([pytorch.org](http://pytorch.org))
- `pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders
    - To do this, you can use the following script: https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh

## run
bash run_image_classify ${batchSize} ${dataDir}
