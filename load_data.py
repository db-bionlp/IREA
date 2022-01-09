# /usr/bin/env python
# coding=utf-8
"""Dataloader"""
import os
import json
import torch
import logging
logger = logging.getLogger(__name__)
import random
from multiprocessing import Pool
import functools
import numpy as np
from collections import defaultdict
from itertools import chain
from utils import Label2IdxSub, Label2IdxObj, _get_so_head
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer

class InputExample(object):
    """a single set of samples of dataset
    """
    def __init__(self, text, en_pair_list, re_list, rel2ens):
        self.text = text
        self.en_pair_list = en_pair_list
        self.re_list = re_list
        self.rel2ens = rel2ens

class InputFeatures(object):
    """
    Desc:
        a single set of features of dataset
    """

    def __init__(self,
                 input_tokens,
                 input_ids,
                 attention_mask,
                 seq_tag=None,
                 corres_tag=None,
                 relation=None,
                 triples=None,
                 rel_tag=None
                 ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.seq_tag = seq_tag
        self.corres_tag = corres_tag
        self.relation = relation
        self.triples = triples
        self.rel_tag = rel_tag

def read_examples(data_dir, data_sign, rel2idx):
    """load dataset to InputExamples
    """
    examples = []

    # read src dataset
    dir= os.path.join(data_dir, '{}_triples.json'.format(data_sign))
    with open(dir, "r", encoding='utf-8') as f:
        data = json.load(f)
        for sample in data:
            text = sample['text']
            rel2ens = defaultdict(list)
            en_pair_list = []
            re_list = []

            if 'triple_list' in sample.keys():
                for triple in sample['triple_list']:
                    en_pair_list.append([triple[0], triple[-1]])
                    re_list.append(rel2idx[triple[1]])
                    rel2ens[rel2idx[triple[1]]].append((triple[0], triple[-1]))
                example = InputExample(text=text, en_pair_list=en_pair_list, re_list=re_list, rel2ens=rel2ens)
                examples.append(example)
    return examples

class FeatureDataset(Dataset):
    """Pytorch Dataset for InputFeatures
    """

    def __init__(self, features):
        self.features = features

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index]

class CustomDataLoader(object):
    def __init__(self, args):
        self.args = args

        self.train_batch_size = args.train_batch_size
        self.val_batch_size = args.val_batch_size
        self.test_batch_size = args.val_batch_size

        self.data_dir = args.data_dir
        self.max_seq_length = args.max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir, do_lower_case=False)

    @staticmethod
    def collate_fn_train(features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        seq_tags = torch.tensor([f.seq_tag for f in features], dtype=torch.long)
        poten_relations = torch.tensor([f.relation for f in features], dtype=torch.long)
        corres_tags = torch.tensor([f.corres_tag for f in features], dtype=torch.long)
        rel_tags = torch.tensor([f.rel_tag for f in features], dtype=torch.long)
        tensors = [input_ids, attention_mask, seq_tags, poten_relations, corres_tags, rel_tags]
        return tensors

    @staticmethod
    def collate_fn_test(features):
        input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        triples = [f.triples for f in features]
        input_tokens = [f.input_tokens for f in features]
        tensors = [input_ids, attention_mask, triples, input_tokens]
        return tensors

    def get_features(self, data_sign):

        cache_path = os.path.join(self.data_dir,
                                  "cached_{}_{}_{}".format(list(filter(None, self.args.bert_model_dir.split("/"))).pop(),
                                                           data_sign, str(self.max_seq_length), self.args.train_batch_size))
        if os.path.exists(cache_path):
            logger.info("loading dataset from {}".format(cache_path))
            features = torch.load(cache_path)
            logger.info("Loaded {} {} dataset(sentences)...".format(len(features), data_sign))
        else:
            # get relation to idx
            dir=os.path.join(self.data_dir,'rel2id.json' )
            with open(dir, 'r', encoding='utf-8') as f_re:
                rel2idx = json.load(f_re)[-1]
            # get examples
            if data_sign in ("train", "val", "test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
                examples = read_examples(self.data_dir, data_sign=data_sign, rel2idx=rel2idx)
            else:
                raise ValueError("please notice that the dataset can only be train/val/test!!")
            features = convert_examples_to_features(self.args, examples, self.tokenizer, rel2idx, data_sign,
                                                    )
            logger.info("maked {} {} dataset(sentences)...".format(len(features), data_sign))
            # save dataset
            torch.save(features, cache_path)
        return features

    def get_dataloader(self, data_sign=None):
        """construct dataloader
        :param data_sign: 'train', 'val' or 'test'
        """
        # InputExamples to InputFeatures
        features = self.get_features(data_sign=data_sign,)
        dataset = FeatureDataset(features)
        if data_sign == "train":
            datasampler = RandomSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.train_batch_size,
                                    collate_fn=self.collate_fn_train)
        elif data_sign == "val":
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.val_batch_size,
                                    collate_fn=self.collate_fn_test)
        elif data_sign in ("test", "pseudo", 'EPO', 'SEO', 'SOO', 'Normal', '1', '2', '3', '4', '5'):
            datasampler = SequentialSampler(dataset)
            dataloader = DataLoader(dataset, sampler=datasampler, batch_size=self.test_batch_size,
                                    collate_fn=self.collate_fn_test)
        else:
            raise ValueError("please notice that the dataset can only be train/val/test !!")
        return dataloader

def convert(example, max_text_len, tokenizer, rel2idx, data_sign):
    text_tokens = tokenizer.tokenize(example.text)
    # cut off
    if len(text_tokens) > max_text_len-2:
        text_tokens = text_tokens[:max_text_len-2]

    text_tokens = ["CLS"] + text_tokens + ["SEP"]

    # token to id
    input_ids = tokenizer.convert_tokens_to_ids(text_tokens)
    attention_mask = [1] * len(input_ids)
    # zero-padding up to the sequence length
    if len(input_ids) < max_text_len:
        pad_len = max_text_len - len(input_ids)
        # token_pad_id=0
        input_ids += [0] * pad_len
        attention_mask += [0] * pad_len

    # train dataset
    if data_sign == 'train':
        # construct tags of correspondence and relation
        corres_tag = np.zeros((max_text_len, max_text_len))
        rel_tag = len(rel2idx) * [0]
        for en_pair, rel in zip(example.en_pair_list, example.re_list):
            # get sub and obj head
            sub_head, obj_head, _, _ = _get_so_head(en_pair, tokenizer, text_tokens)
            # construct relation tag
            rel_tag[rel] = 1
            if sub_head != -1 and obj_head != -1:
                corres_tag[sub_head][obj_head] = 1

        sub_feats = []
        # positive samples
        for rel, en_ll in example.rel2ens.items():
            # init
            tags_sub = max_text_len * [Label2IdxSub['O']]
            tags_obj = max_text_len * [Label2IdxSub['O']]
            for en in en_ll:
                # get sub and obj head
                sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens)
                if sub_head != -1 and obj_head != -1:
                    if sub_head + len(sub) <= max_text_len:
                        tags_sub[sub_head] = Label2IdxSub['B-H']
                        tags_sub[sub_head + 1:sub_head + len(sub)] = (len(sub) - 1) * [Label2IdxSub['I-H']]
                    if obj_head + len(obj) <= max_text_len:
                        tags_obj[obj_head] = Label2IdxObj['B-T']
                        tags_obj[obj_head + 1:obj_head + len(obj)] = (len(obj) - 1) * [Label2IdxObj['I-T']]
            seq_tag = [tags_sub, tags_obj]

            # sanity check
            assert len(input_ids) == len(tags_sub) == len(tags_obj) == len(
                attention_mask) == max_text_len, f'length is not equal!!'

            sub_feats.append(InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                corres_tag=corres_tag,
                seq_tag=seq_tag,
                relation=rel,
                rel_tag=rel_tag
            ))
    # val and test dataset
    else:
        triples = []
        for rel, en in zip(example.re_list, example.en_pair_list):
            # get sub and obj head
            sub_head, obj_head, sub, obj = _get_so_head(en, tokenizer, text_tokens)
            if sub_head != -1 and obj_head != -1:
                h_chunk = ('H', sub_head, sub_head + len(sub))
                t_chunk = ('T', obj_head, obj_head + len(obj))
                triples.append((h_chunk, t_chunk, rel))

        sub_feats = [
            InputFeatures(
                input_tokens=text_tokens,
                input_ids=input_ids,
                attention_mask=attention_mask,
                triples=triples
            )
        ]

    # get sub-feats
    return sub_feats

def convert_examples_to_features(args, examples, tokenizer, rel2idx, data_sign):
    """convert examples to features.
    :param examples (List[InputExamples])
    """
    max_text_len = args.max_seq_length
    # multi-process
    with Pool(10) as p:
        convert_func = functools.partial(convert, max_text_len=max_text_len, tokenizer=tokenizer, rel2idx=rel2idx,
                                         data_sign=data_sign, )
        features = p.map(func=convert_func, iterable=examples)

    return list(chain(*features))