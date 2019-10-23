# No. DC-AI-C13: 3D object reconstruction
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This benchmark predicts and reconstructs 3D objects.
It uses a convolutional encoder-decoder network and takes **ShapeNet Dataset** as input.

This code requires the dataset to be in *tfrecords* format with the following features:
*   image
    *   Flattened list of image (float representations) for each view point.
*   mask
    *   Flattened list of image masks (float representations) for each view point.
*   vox
    *   Flattened list of voxels (float representations) for the object.
    *   This is needed for using vox loss and for prediction comparison.

You can download the ShapeNet Dataset in tfrecords format from [here](https://drive.google.com/file/d/0B12XukcbU7T7OHQ4MGh6d25qQlk)

## How to run

### Training

```bash
python train_ptn.py --input_dir ./dataset
```
