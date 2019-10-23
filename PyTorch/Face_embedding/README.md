# No. DC-AI-C7: Face embedding
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This benchmark transforms a facial image to a vector in embedding space. It uses the FaceNet algorithm and takes the LFW (Labeled Faces in the Wild) dataset or VGGFace2 as input.


## How to run

### Preprocessing
#### Face alignment


```Shell
cd ${facenet}
../scripts/face-align-CASIA-Webface.sh
../scripts/face-align-VGGface2.sh   # This might takes several hours
../scripts/face-align-LFW.sh
```

#### Generating csv files  

```Python
which_dataset = 1
```
```Python
if   which_dataset == 0:
    root_dir = "/run/media/hoosiki/WareHouse2/home/mtb/datasets/vggface2/train_mtcnnpy_182"
elif which_dataset == 1:
    root_dir = "/run/media/hoosiki/WareHouse2/home/mtb/datasets/vggface2/test_mtcnnpy_182"
elif which_dataset == 2:
    root_dir = "/run/media/hoosiki/WareHouse2/home/mtb/datasets/lfw/lfw_mtcnnpy_182"
else:
    root_dir = "/run/media/hoosiki/WareHouse2/home/mtb/datasets/my_pictures/my_pictures_mtcnnpy_182"
```
```Shell
python datasets/write_csv_for_making_dataset.py
```

### Training

```Shell
cd ${facenet}
./run.sh
```

