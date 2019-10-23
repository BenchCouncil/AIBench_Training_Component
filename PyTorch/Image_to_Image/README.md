# No. DC-AI-C5: Image-to-Image
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This problem domain is to convert an image from one representation of an image to another representation.

## 准备工作
### 环境搭建
```Shell
cd ${pytorch-CycleGAN}
pip install -r requirements.txt
```

### 下载数据集

```Shell
arr=("ae_photos" "apple2orange" "summer2winter_yosemite"  "horse2zebra" \
    "monet2photo" "cezanne2photo" "ukiyoe2photo" "vangogh2photo" "maps" \
    "cityscapes" "facades" "iphone2dslr_flower" "mini" "mini_pix2pix" \
    "mini_colorization")
# Download a CycleGAN dataset (e.g. maps):
bash ./datasets/download_cyclegan_dataset.sh cityscapes

```

## 用法
参考[这里](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#cyclegan-traintest)


## 参考链接
1. [CycleGAN Paper](https://arxiv.org/abs/1703.10593.pdf)
1. [CycleGAN Project Homepage](https://junyanz.github.io/CycleGAN/)
1. [https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
