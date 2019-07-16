# 2. 图像生成 WGAN (PyTorch)

## 准备工作

### 下载数据集
```Shell
cd ${lsun}

# Download data for bedroom set to <data_dir>
python3 download.py -o <data_dir> -c bedroom
```

## 用法
**With DCGAN:**

```bash
python main.py --dataset lsun --dataroot [lsun-train-folder] --cuda
```

**With MLP:**

```bash
python main.py --mlp_G --ngf 512 --dataset lsun --dataroot [lsun-train-folder] --cuda
```

Generated samples will be in the `samples` folder.

If you plot the value `-Loss_D`, then you can reproduce the curves from the paper. The curves from the paper (as mentioned in the paper) have a median filter applied to them:

```python
med_filtered_loss = scipy.signal.medfilt(-Loss_D, dtype='float64'), 101)
```

# 遇到的一些问题
> 1. Traceback (most recent call last):
>   File "main.py", line 81, in <module>
>     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
> TypeError: __init__() got an unexpected keyword argument 'db_path'
> **Solution**: [https://github.com/martinarjovsky/WassersteinGAN/issues/66](https://github.com/martinarjovsky/WassersteinGAN/issues/66)


## 参考链接
1. [https://github.com/martinarjovsky/WassersteinGAN](https://github.com/martinarjovsky/WassersteinGAN)
1. [Paper: Wasserstein GAN](https://arxiv.org/abs/1701.07875)
1. [https://github.com/fyu/lsun](https://github.com/fyu/lsun)
1. [LSUN Homepage](https://www.yf.io/p/lsun)
1. [LSUN: Construction of a Large-scale Image Dataset using Deep Learning with Humans in the Loop](https://arxiv.org/abs/1506.03365)
