# No. DC-AI-C8: 3D face recognition
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This benchmark recognize the 3D facial information from an image. It provides a series of 3D face modes. The test data is provided by Intellifusion and includes 77,715 samples from 253 face IDs, which are published on the BenchCouncil web site.

Dataset is available from [here](http://125.39.136.212:8484/3dvggface2_1.tar.gz).

## How to run

### Preprocessing
#### Face alignment
```bash
export PYTHONPATH=${PWD}/src
python preprocess/align/align_dataset_mtcnn.py \
    --input_dir /mnt/sdb/vggface3 \
    --output_dir /mnt/sdb/vggface3_align \
    --image_size 182 \
    --margin 44 \
    --random_order \
    --thread_num 3 \
    --gpu_memory_fraction 0.88
```
#### Dataset splitting
To split the whole dataset randomly into 3 sub-datasets, (i.e., training dataset, evaluation dataset, test dataset), by generating 3 corresponding csv files to record the paths and labels of each images.
```bash
python preprocess/get_dataset_csv.py
```

