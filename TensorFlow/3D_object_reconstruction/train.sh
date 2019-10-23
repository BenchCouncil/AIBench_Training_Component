
# please confirm download dataset has been downloaded
# from https://drive.google.com/file/d/0B12XukcbU7T7OHQ4MGh6d25qQlk
export PYTHONPATH=$(cd `dirname $0` && pwd):$PYTHONPATH
echo $PYTHONPATH
python train_ptn.py --input_dir ./dataset
