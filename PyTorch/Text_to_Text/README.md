# 3. Text-to-Text Translation Pytorch Version

## Qucik Start:
You can run
```shell
sh download.sh
```
to download a small dataset.
```
sh run.sh
```
to train.
## Or train your own model:
```
cd run-attention-is-all-you-need
```
### 1)Download your own dataset
### 2)Preprocess the data
```
python preprocess.py -train_src $your_train_data_language1 -train_tgt $your_trian_data_language2 -valid_src $your_valid_data_language1 -valid_tgt $your_valid_data_language2 -save_data $your_data_saved.pt
```
### 3)Train the model
```
python train.py -data $your_data_saved.pt -save_model trained -save_mode best -proj_share_weight -label_smoothing
```
### 4)Test the model 
```
python translate.py -model trained.chkpt -vocab $your_data_saved.pt -src $your_test_data
```
## Reference
https://github.com/jadore801120/attention-is-all-you-need-pytorch
