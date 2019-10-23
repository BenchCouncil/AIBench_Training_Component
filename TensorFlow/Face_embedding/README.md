# No. DC-AI-C7: Face embedding
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This benchmark transforms a facial image to a vector in embedding space. It uses the FaceNet algorithm and takes the LFW (Labeled Faces in the Wild) dataset or VGGFace2 as input.

## 准备工作

**Note**: 下载预训练模型、数据集之后，需要对数据集进行 MTCNN 剪裁，并在 `${facenet}` 目录下生成 models, logs, datasets 的软链接。

### 下载预训练模型
参考 [davidsandberg/facenet#pre-trained-models](https://github.com/davidsandberg/facenet#pre-trained-models)

| Model name      | LFW accuracy | Training dataset | Architecture |
|-----------------|--------------|------------------|-------------|
| [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) | 0.9905        | CASIA-WebFace    | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |
| [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-) | 0.9965        | VGGFace2      | [Inception ResNet v1](https://github.com/davidsandberg/facenet/blob/master/src/models/inception_resnet_v1.py) |

> **Note**: The above table is quoted from [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet/blob/master/README.md#pre-trained-models)

```Shell
cd ${facenet}
```

- ** 通过 Python 下载: **
```Python
# url: https://drive.google.com/file/d/1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz/view
from src.download_and_extract import download_file_from_google_drive
download_file_from_google_drive('1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz', '/your/path/to/20180408-102900.zip')
download_file_from_google_drive('1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-', '/your/path/to/20180402-114759.zip')
```

- ** 通过 bash (wget) 下载: [推荐使用] **
```Shell
../scripts/gdriver-download.sh 1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz /your/path/to/20180408-102900.zip
../scripts/gdriver-download.sh 1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55- /your/path/to/20180402-114759.zip
```

> **Note**: Please make sure you can access google!

### 下载数据集
#### CASIA-WebFace 下载

dataset size: 4.1G
Google Drive fileID: `1Of_EVz-yHV7QVWQGihYfvtny9Ne8qXVz`

下载方法参考上面 [下载预训练模型](#下载预训练模型)

#### VGGFace2 下载

training set size: 36G
test set size: 1.9G

```Shell
cd ${facenet}
../scripts/vggface2-download.py
mv ../scripts/vggface2_[train|test].tar.gz /your/path/to/vggface2/
```

### 图像裁剪
```Shell
cd ${facenet}
../scripts/face-align-CASIA-Webface.sh
../scripts/face-align-VGGface2.sh   # This might takes several hours
../scripts/face-align-LFW.sh
```

## 用法

### 训练 & 验证
```Shell
cd ${facenet}

# Train on the CASIA-Webface dataset with the `softmax loss`
# And, validate on the LFW dataset
../scripts/cls_training_softmax_webface.sh

# Train on the VGGFace2 dataset with the `softmax loss`
# And, validate on the LFW dataset
../scripts/cls_training_softmax_vggface2.sh

# Trainning on CASIA-Webface dataset with the `triplet loss`
# And, validate on the LFW dataset
../scripts/cls_training_triplet_webface.sh
```

### 验证

```Shell
cd ${facenet}

# Validate on the LFW dataset
../scripts/validate_on_lfw.sh
```

## 参考链接
1. [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)
1. [Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." *Proceedings of the IEEE conference on computer vision and pattern recognition.*2015.](https://arxiv.org/abs/1503.03832)
1. [Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. VGGFace2: A dataset for recognising face across pose and age, International Conference on Automatic Face and Gesture Recognition, 2018.](http://zeus.robots.ox.ac.uk/vgg_face2)
