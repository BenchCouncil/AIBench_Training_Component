# Run the training.
python seq2seq_attention.py \
    --mode=train \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/data \
    --vocab_path=data/vocab \
    --log_root=textsum/log_root \
    --train_dir=textsum/log_root/train

