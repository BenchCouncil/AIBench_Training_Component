# `python train_softmax.sh -h` for help.
python train_softmax.sh \
    --train_dataset_csv ~/vggface3d_sm/train.csv \
    --eval_dataset_csv ~/vggface3d_sm/eval.csv \
    --num_of_workers 16 
