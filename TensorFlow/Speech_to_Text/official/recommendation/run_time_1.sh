rm -r /tmp/ncf*
export PYTHONPATH="/home/gwl/tensorflow/tran_base/Speech_to_Text/deep_speech2/"
CUDA_VISIBLE_DEVICES=0 python ncf_main.py \
  --data_dir /home/gwl/tensorflow/dataset/recommendation/movielens-data/ --num_gpus 1
