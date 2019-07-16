batch_size=$1
data_dir=$2

python main.py -a resnet50 -b ${batch_size} ${data_dir}
