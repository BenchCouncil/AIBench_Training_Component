# No. DC-AI-C11: Video prediction
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This benchmark predicts the future video through predicting previous frames transformation. It uses motion-focused predictive models and takes **Robot pushing dataset** as input.


To download the data, run the following.



## Datasets

### Moving MNIST
Download the original [MNIST](http://yann.lecun.com/exdb/mnist/) dataset and the [Moving MNIST test set](http://www.cs.toronto.edu/~nitish/unsupervised_video/) by running
```
./datasets/moving_mnist/download.sh
```

### Bouncing Balls
We generate our Bouncing Balls dataset with the [Neural Physics Engine](https://github.com/mbchang/dynamics), used by Chang et al, 2017.

First, clone the [bouncing_balls](https://github.com/jthsieh/dynamics) submodule,
```
git submodule update --init --recursive
```
Note that this is slightly modified from the original [dynamics](https://github.com/mbchang/dynamics) repository.

`npm` must be installed. With Anaconda, install by running: `conda install -c conda-forge nodejs`.
Then,
```
cd datasets/bouncing_balls/src/js/
npm install
```

Generate training and testing data,
```
node demo/js/generate.js -e balls -n 4 -t 60 -s 50000
node demo/js/generate.js -e balls -n 4 -t 60 -s 2000
```

This will generate two folders, `balls_n4_t60_ex50000` and `balls_n4_t60_ex2000`, in the `datasets/bouncing_balls/data/` directory.
Move these folders to the desired location (dataset path).
Next, modify the `root` variable in `datasets/bouncing_balls/process.py` and run `python datasets/bouncing_balls/process.py` for both files.



## How to run

### Training

```bash
./train.sh
```
