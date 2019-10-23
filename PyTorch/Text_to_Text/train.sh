#!/bin/bash

python attention-is-all-you-need-pytorch/preprocess.py -train_src  ../../DataSet/WMT-English-German/training/train.en -train_tgt  ../../DataSet/WMT-English-German/training/train.de -valid_src  ../../DataSet/WMT-English-German/validation/val.en -valid_tgt ../../DataSet/WMT-English-German/validation/val.de -save_data ../../DataSet/WMT-English-German/multi30k.atok.low.pt
python attention-is-all-you-need-pytorch/train.py -data ../../DataSet/WMT-English-German/multi30k.atok.low.pt -save_model trained -save_mode best -proj_share_weight -label_smoothing -batch_size 256 -epoch 20
python attention-is-all-you-need-pytorch/translate.py -model trained.chkpt -vocab ../../DataSet/WMT-English-German/multi30k.atok.low.pt -src ../../DataSet/WMT-English-German/testing/test.en
