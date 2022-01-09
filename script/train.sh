python3 ../trainer.py \
--epoch_num=250 \
--device_id=1 \
--max_seq_length=100 \
--corpus_type DDI \
--ensure_corres \
--ensure_rel \
--bert_model_dir allenai/scibert_scivocab_cased \
--model_dir ../saved_model/ddi_cased_len100_ep250

#--multi_gpu \