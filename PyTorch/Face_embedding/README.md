# 7. 人脸识别 FaceNet (PyTorch)

## 准备工作

**Note**: 需要先运行 `TensorFlow` 版本的 `FaceNet` ，按照前述环境下载数据集并剪裁图片。
**Warning**: PyTorch对数据集尺寸要求严格，请按照下述方法修改代码。

### 图像裁剪

`PyTorch` 对图像尺寸要求比较严格，请将 `/scripts/face-align-CASIA-Webface.sh` 中的参数  
`--image_size 182`
改为  
`--image_size 224`
重新运行剪裁图像。
```Shell
cd ${facenet}
../scripts/face-align-CASIA-Webface.sh
../scripts/face-align-VGGface2.sh   # This might takes several hours
../scripts/face-align-LFW.sh
```

## 用法

### 生成csv文件

重写文件 `datasets/write_csv_for_making_dataset.py` ，你需要更改参数 `which_dataset` 以及文件路径 `root_dir` 。
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
修改并运行四次文件 `datasets/write_csv_for_making_dataset.py` ，生成四个csv文件。
```Shell
python datasets/write_csv_for_making_dataset.py
```

### 训练 & 验证

运行脚本
```Shell
cd ${facenet}
./run.sh
```

## 参考链接
1. [https://github.com/tbmoon/facenet](https://github.com/tbmoon/facenet)
2. [Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." *Proceedings of the IEEE conference on computer vision and pattern recognition.*2015.](https://arxiv.org/abs/1503.03832)

