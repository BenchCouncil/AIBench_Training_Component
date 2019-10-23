# No. DC-AI-C4: Image-to-Text
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This problem domain is to generate the description of an image automatically.


## 准备工作
### 环境搭建
1. Bazel
2. Natural Language Toolkit (NLTK)
3. unzip, numpy

### 下载预训练模型 (Inception V3)
见 [https://github.com/tensorflow/models/tree/master/research/im2txt#download-the-inception-v3-checkpoint](https://github.com/tensorflow/models/tree/master/research/im2txt#download-the-inception-v3-checkpoint)

### 下载数据集 (Coco2014)
见 [https://github.com/tensorflow/models/tree/master/research/im2txt#prepare-the-training-data](https://github.com/tensorflow/models/tree/master/research/im2txt#prepare-the-training-data)

## 用法
```Shell
IM2TXT_HOME=/mnt/sdb/xingw

# Directory containing preprocessed MSCOCO data.
MSCOCO_DIR="${IM2TXT_HOME}/im2txt/data/mscoco"

# Inception v3 checkpoint file.
INCEPTION_CHECKPOINT="${IM2TXT_HOME}/im2txt/data/inception_v3.ckpt"

# Directory to save the model.
MODEL_DIR="${IM2TXT_HOME}/im2txt/model"

# Build the model.
cd research/im2txt
bazel build -c opt //im2txt/...

# Run the training script.
bazel-bin/im2txt/train \
  --input_file_pattern="${MSCOCO_DIR}/train-?????-of-00256" \
  --inception_checkpoint_file="${INCEPTION_CHECKPOINT}" \
  --train_dir="${MODEL_DIR}/train" \
  --train_inception=false \
  --number_of_steps=1000000
```

## 遇到的一些问题
**1. 将 Python2 代码转化为 Python3。**  
```Shell
2to3 -w /usr/local/lib/python3.6/dist-packages/tensorflow/models/research/im2txt 
```

**2. tf.gfile.FastGFile(filename, 'r').read() error: 'utf-8' codec can't decode byte 0xff**   
> 修改 `research/im2txt/im2txt/data/build_mscoco_data.py` 文件，把 `tf.gfile.FastGFile(image_filename, "r")` 修改为 `tf.gfile.FastGFile(image_filename, "rb")` 

**3. tf.train.byteslist has type str but expected one of bytes**  
> 修改 `research/im2txt/im2txt/data/build_mscoco_data.py` 文件:   
```Python
def _bytes_feature(value):
  """Wrapper for inserting a bytes Feature into a SequenceExample proto."""
  value=tf.compat.as_bytes(value)
  # return tf.train.Feature(bytes_list=tf.train.BytesList(value=[str(value)]))
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
```

**4. INFO:tensorflow:Error reported to Coordinator: <class 'tensorflow.python.framework.errorsimpl.DataLossError'>, truncated reco**  
> 这种情况一般都是数据有损坏，如果是tfrecord的数据，建议重新生成，要是直接是图片数据，那就需要好好检查了。实在不行，建议用MD5做下文件校验。  


## 参考链接
1. [Source Code: tensorflow/model](https://github.com/tensorflow/models/tree/master/research/im2txt)
