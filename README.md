# A joint drug-drug interaction relationship extraction framework based on implicit relationships and entity alignment

## Overview

 ![image](img/IREA_model_.png)

## Requirements

The main requirements are:

  - python==3.6.9
  - pytorch==1.7.0
  - transformers==3.2.0
  - tqdm

## Datasets

- [NYT*](https://github.com/weizhepei/CasRel/tree/master/data/NYT) and [WebNLG*](https://github.com/weizhepei/CasRel/tree/master/data/WebNLG)(following [CasRel](https://github.com/weizhepei/CasRel))
- [NYT](https://drive.google.com/file/d/1kAVwR051gjfKn3p6oKc7CzNT9g2Cjy6N/view)(following [CopyRE](https://github.com/xiangrongzeng/copy_re))
- [WebNLG](https://github.com/yubowen-ph/JointER/tree/master/dataset/WebNLG/data)(following [ETL-span](https://github.com/yubowen-ph/JointER))

Or you can just download our preprocessed [datasets](https://drive.google.com/file/d/1hpUedGxzpg6lyNemClfMCeTXeaBBQ1u7/view?usp=sharing).

## Usage

**1. Build Data**

Put our preprocessed datasets under `./datasets`.

**2. Train**

Just run the script in `./script` by `sh train.sh`.

For example, to train the model for DDI dataset, update the `train.sh` as:

```
python3 main.py \
--data_dir dataset/DDI_2013 \
--epoch_num=100 \
--max_seq_length=128 \
--bert_model_dir allenai/scibert_scivocab_cased \
--train_batch_size 16 \
--val_batch_size 16 \
--model_dir saved_model/DDI_2013
```

