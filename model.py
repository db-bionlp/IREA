# /usr/bin/env python
# coding=utf-8
"""model"""
from collections import Counter
import torch,os
import torch.nn as nn
import json
Label2IdxSub = {"B-H": 1, "I-H": 2, "O": 0}
Label2IdxObj = {"B-T": 1, "I-T": 2, "O": 0}

from transformers import BertPreTrainedModel, BertModel

class MultiNonLinearClassifier(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(MultiNonLinearClassifier, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        features_output = self.hidden2tag(features_tmp)
        return features_output

class SequenceLabelForSO(nn.Module):
    def __init__(self, hidden_size, tag_size, dropout_rate):
        super(SequenceLabelForSO, self).__init__()
        self.tag_size = tag_size
        self.linear = nn.Linear(hidden_size, int(hidden_size / 2))
        self.hidden2tag_sub = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.hidden2tag_obj = nn.Linear(int(hidden_size / 2), self.tag_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input_features):
        """
        Args:
            input_features: (bs, seq_len, h)
        """
        features_tmp = self.linear(input_features)
        features_tmp = nn.ReLU()(features_tmp)
        features_tmp = self.dropout(features_tmp)
        sub_output = self.hidden2tag_sub(features_tmp)
        obj_output = self.hidden2tag_obj(features_tmp)
        return sub_output, obj_output

class BertForRE(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)
        self.max_seq_len = args.max_seq_length
        self.seq_tag_size = len(Label2IdxSub)
        self.rel2idx = json.load(open(os.path.join(args.data_dir, 'rel2id.json'), 'r', encoding='utf-8'))[-1]
        self.rel_num = len(self.rel2idx)
        self.args = args


        self.bert = BertModel(config)
        # sequence tagging
        self.sequence_tagging_sub = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, args.drop_prob)
        self.sequence_tagging_obj = MultiNonLinearClassifier(config.hidden_size * 2, self.seq_tag_size, args.drop_prob)
        self.sequence_tagging_sum = SequenceLabelForSO(config.hidden_size, self.seq_tag_size, args.drop_prob)

        self.expansion = MultiNonLinearClassifier(self.seq_tag_size, config.hidden_size, args.drop_prob)
        self.global_corres = MultiNonLinearClassifier(config.hidden_size * 2, 1, args.drop_prob)

        self.rel_judgement = MultiNonLinearClassifier(config.hidden_size*3, self.rel_num, args.drop_prob)
        self.rel_embedding = nn.Embedding(self.rel_num, config.hidden_size)

        self.init_weights()

    @staticmethod
    def masked_avgpool(sent, mask):
        mask_ = mask.masked_fill(mask == 0, -1e9).float()
        score = torch.softmax(mask_, -1)
        return torch.matmul(score.unsqueeze(1), sent).squeeze(1)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            seq_tags=None,
            potential_rels=None,
            corres_tags=None,
            rel_tags=None,
    ):

        # get params for experiments
        global rel_pred
        Alignment_threshold, Rel_threshold = self.args.Alignment_threshold, self.args.Rel_threshold

        # pre-train model
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        sequence_output = outputs[0]
        avg_pooled_output = self.masked_avgpool(sequence_output, attention_mask)
        first_pooled_output = outputs[1]
        last_pooled_output = sequence_output[:, - 1]
        bs, seq_len, h = sequence_output.size()
        sequence = torch.cat([avg_pooled_output, first_pooled_output, last_pooled_output], dim=-1)
        rel_pred = self.rel_judgement(sequence)

        # for every position $i$ in sequence, should concate $j$ to predict.
        sub_extend = sequence_output.unsqueeze(2).expand(-1, -1, seq_len, -1)  # (bs, s, s, h)
        obj_extend = sequence_output.unsqueeze(1).expand(-1, seq_len, -1, -1)  # (bs, s, s, h)
        # batch x seq_len x seq_len x 2*hidden
        corres_pred = torch.cat([sub_extend, obj_extend], 3)
        # (bs, seq_len, seq_len)
        corres_pred = self.global_corres(corres_pred).squeeze(-1)
        mask_tmp1 = attention_mask.unsqueeze(-1)
        mask_tmp2 = attention_mask.unsqueeze(1)
        corres_mask = mask_tmp1 * mask_tmp2

        # relation predict and dataset construction in inference stage
        xi, pred_rels = None, None
        if seq_tags is None:
            rel_pred_onehot = torch.where(torch.sigmoid(rel_pred).to(self.args.device) > Rel_threshold,
                                          torch.ones(rel_pred.size(), device=self.args.device),
                                          torch.zeros(rel_pred.size(), device=self.args.device))
            #there is no rel pred
            for idx, sample in enumerate(rel_pred_onehot):
                if 1 not in sample:
                    max_index = torch.argmax(rel_pred[idx])
                    sample[max_index] = 1
                    rel_pred_onehot[idx] = sample


            bs_idxs, pred_rels = torch.nonzero(rel_pred_onehot, as_tuple=True)
            # get x_i
            xi_dict = Counter(bs_idxs.tolist())
            xi = [xi_dict[idx] for idx in range(bs)] #每个句子多少rel
            xi = torch.tensor(xi).to(input_ids.device)

            pos_seq_output = []
            pos_potential_rel = []
            pos_attention_mask = []
            for bs_idx, rel_idx in zip(bs_idxs, pred_rels):
                # (seq_len, h)
                pos_seq_output.append(sequence_output[bs_idx])
                pos_attention_mask.append(attention_mask[bs_idx])
                pos_potential_rel.append(rel_idx)
            # (sum(x_i), seq_len, h)
            sequence_output = torch.stack(pos_seq_output, dim=0).to(input_ids.device)
            # (sum(x_i), seq_len)
            attention_mask = torch.stack(pos_attention_mask, dim=0).to(input_ids.device)
            # (sum(x_i),)
            potential_rels = torch.stack(pos_potential_rel, dim=0).to(input_ids.device)

        rel_emb = self.rel_embedding(potential_rels)

        # relation embedding vector fusion
        rel_emb = rel_emb.unsqueeze(1).expand(-1, seq_len, h)

        decode_input = torch.cat([sequence_output, rel_emb], dim=-1)
        output_sub = self.sequence_tagging_sub(decode_input)
        output_obj = self.sequence_tagging_obj(decode_input)


        # sequence_sub = self.expansion(output_sub)
        # sqquence_obj = self.expansion(output_obj)
        # sub_extend = sequence_sub.unsqueeze(2).expand(-1, -1, seq_len, -1)
        # obj_extend = sqquence_obj.unsqueeze(1).expand(-1, seq_len, -1, -1)
        #
        # corres_pred = torch.cat([sub_extend, obj_extend], 3)
        #
        #     # (bs, seq_len, seq_len)
        # corres_pred = self.global_corres(corres_pred).squeeze(-1)
        # mask_tmp1 = attention_mask.unsqueeze(-1)
        # mask_tmp2 = attention_mask.unsqueeze(1)
        # corres_mask = mask_tmp1 * mask_tmp2


        # train
        if seq_tags is not None:
            # calculate loss
            attention_mask = attention_mask.view(-1)
            # sequence label loss
            loss_func = nn.CrossEntropyLoss(reduction='none')
            loss_seq_sub = (loss_func(output_sub.view(-1, self.seq_tag_size),
                                      seq_tags[:, 0, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            loss_seq_obj = (loss_func(output_obj.view(-1, self.seq_tag_size),
                                      seq_tags[:, 1, :].reshape(-1)) * attention_mask).sum() / attention_mask.sum()
            loss_seq = (loss_seq_sub + loss_seq_obj) / 2
            loss_seq=loss_seq.to(self.args.device)
            # init
            loss_matrix, loss_rel = torch.tensor(0).to(self.args.device), torch.tensor(0).to(self.args.device)
            corres_pred = corres_pred.view(bs, -1).to(self.args.device)
            corres_mask = corres_mask.view(bs, -1).to(self.args.device)
            corres_tags = corres_tags.view(bs, -1)
            loss_func = nn.BCEWithLogitsLoss(reduction='none')
            loss_matrix = (loss_func(corres_pred.to(self.args.device),
                                         corres_tags.float().to(self.args.device)) * corres_mask).sum() / corres_mask.sum()

            loss_func = nn.BCEWithLogitsLoss(reduction='mean')

            loss_rel = loss_func(rel_pred.to(self.args.device), rel_tags.float().to(self.args.device))


            loss = 2*loss_seq + loss_matrix + loss_rel
            return loss, loss_seq, loss_matrix, loss_rel
        # inference
        else:
            # (sum(x_i), seq_len)
            pred_seq_sub = torch.argmax(torch.softmax(output_sub, dim=-1), dim=-1)
            pred_seq_obj = torch.argmax(torch.softmax(output_obj, dim=-1), dim=-1)
            # (sum(x_i), 2, seq_len)
            pred_seqs = torch.cat([pred_seq_sub.unsqueeze(1), pred_seq_obj.unsqueeze(1)], dim=1)
            corres_pred = torch.sigmoid(corres_pred) * corres_mask
            pred_corres_onehot = torch.where(corres_pred > Alignment_threshold,
                                                 torch.ones(corres_pred.size(), device=corres_pred.device),
                                                 torch.zeros(corres_pred.size(), device=corres_pred.device))
            return pred_seqs, pred_corres_onehot, xi, pred_rels


