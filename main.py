# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, [0, ]))
import argparse
import logging
logger = logging.getLogger(__name__)
from load_data import CustomDataLoader
from utils import init_logger
import torch
from transformers import BertConfig
from  model import BertForRE
from trainer import train_and_evaluate
def main(args):
    init_logger()
    # Load training dataset and val dataset
    dataloader = CustomDataLoader(args)
    train_loader = dataloader.get_dataloader(data_sign='train')
    val_loader = dataloader.get_dataloader(data_sign='val')
    test_loader = dataloader.get_dataloader(data_sign='test')

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logging.info('Load pre-train model weights...')
    bert_config = BertConfig.from_pretrained(args.bert_model_dir)
    model = BertForRE.from_pretrained(config=bert_config,
                                      pretrained_model_name_or_path=args.bert_model_dir,
                                      args=args)

    train_and_evaluate(model, args, args.store_file, train_loader, val_loader, test_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default='dataset/DDI_2013', type=str,
                        help="The input dataset dir. Should contain the .tsv files (or other dataset files) for the task.")
    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
    parser.add_argument("--val_batch_size", default=16, type=int, help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--seed', type=int, default=2020, help="random seed for initialization")
    parser.add_argument('--epoch_num', default=3, type=int, help="number of epochs")
    parser.add_argument('--store_file', default=None, help="name of the file containing weights to reload, Optimum or last")
    parser.add_argument('--model_dir', default="saved_model/DDI_2013", help="saved model")
    parser.add_argument('--bert_model_dir', default='allenai/scibert_scivocab_cased', help="bert model")
    parser.add_argument('--max_seq_length', type=int, default=128, help="max_seq_length")

    parser.add_argument('--Alignment_threshold', type=float, default=0.5, help="threshold of global correspondence")
    parser.add_argument('--Rel_threshold', type=float, default=0.5, help="threshold of relation judgement")
    # learning rate
    parser.add_argument('--fin_tuning_lr', type=float, default=5e-5, help="")
    parser.add_argument('--downs_en_lr', type=float, default=1e-3, help="")
    parser.add_argument('--clip_grad', type=float, default=2., help="")
    parser.add_argument('--drop_prob', type=float, default=0.3, help="")
    parser.add_argument('--weight_decay_rate', type=float, default=0.01, help="")
    parser.add_argument('--warmup_prop', type=float, default=0.1, help="")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2, help="")

    args = parser.parse_args()

    main(args)