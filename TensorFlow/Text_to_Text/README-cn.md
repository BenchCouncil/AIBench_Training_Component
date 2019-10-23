# No. DC-AI-C3: Text-to-Text translation
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This problem domain need to translate text from one language to another, which is the most important field of computational linguistics


## 声明

代码来源于 [tensor2tensor](https://github.com/tensorflow/tensor2tensor.git)，并根据tensorflow的版本和大文件下载进行了一些调整。





## 准备工作
### 环境搭建
`deepo` 镜像中 `tensorflow` 、 `tensorflow-gpu` 的版本为 `1.12.0` , 等，`tensor2tensor` 需要的 `tensorflow` 、 `tensorflow-gpu` 版本为 `1.13.0` ，在未改变 [`docker` 镜像](#) 前需要升级。在进行升级时要注意 `shell` 命令的顺序，由于 `tensorflow-1.13.0` 的版本需要用到 `tensorflow-gpu-1.13.0` 所以先升级后者。

#### 升级

```Shell
pip install --upgrade tensorflow-gpu
pip install --upgrade tensorflow
```

### 下载 [tensor2tensor](https://github.com/tensorflow/tensor2tensor.git)

#### 在 `tensor2tensor` 教程中自带数据下载，如果 `deepo` 所在环境支持 `VPS` 技术可直接按照 [tensor2tensor](https://github.com/tensorflow/tensor2tensor.git) `Basics` 模块教程执行。如果不支持，在第三步
```
t2t-datagen \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR \
  --problem=$PROBLEM

```
#### 会下载四个数据集。
```
http://data.statmt.org/wmt18/translation-task/training-parallel-nc-v13.tgz to /tmp/t2t_datagen/training-parallel-nc-v13.tgz
http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz to /tmp/t2t_datagen/training-parallel-commoncrawl.tgz
http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz to /tmp/t2t_datagen/training-parallel-europarl-v7.tgz
http://data.statmt.org/wmt17/translation-task/dev.tgz to /tmp/t2t_datagen/dev.tgz
```
#### 其中前三个数据集可以在可以直接移到 `/tmp/t2t_datagen/` 中，最后一个文件移动之后，由于函数没有对已存在文件名进行判断，需要修改两个 `.py` 函数。

```Shell
vim /usr/local/lib/python3.6/dist-packages/tensorflow/python/lib/io/file_io.py 
```

#### 其中第496和第512行

```
def rename(oldname, newname, overwrite=False):
def rename_v2(src, dst, overwrite=False):
```
#### 改为
```
def rename(oldname, newname, overwrite=True):
def rename_v2(src, dst, overwrite=True):
```
## 参考链接
1. 暂无
