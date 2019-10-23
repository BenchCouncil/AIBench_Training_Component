# No. DC-AI-C14: Text summarization
This is one of the component benchmarks of AIBench, an Industry Standard AI Benchmark Suite for datacenter AI benchmarking.

This benchmark is to generate the text summary. It uses **sequence-to-sequence** model and takes **Gigaword dataset** as input.

To download the robot data, run the following.

```bash
./download_data.sh
```

## How to run

### Training

```bash
python seq2seq_attention.py \
    --mode=train \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/data \
    --vocab_path=data/vocab \
    --log_root=textsum/log_root \
    --train_dir=textsum/log_root/train
```

### Test

```bash
python seq2seq_attention.py \
    --mode=eval \
    --article_key=article \
    --abstract_key=abstract \
    --data_path=data/data \
    --vocab_path=data/vocab \
    --log_root=textsum/log_root \
    --train_dir=textsum/log_root/eval
```
