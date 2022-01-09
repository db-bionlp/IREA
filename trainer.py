import torch
from transformers import *
import shutil
import logging
logger = logging.getLogger(__name__)
import numpy as np
from tqdm import trange, tqdm
import utils
from optimization import BertAdam
from utils import *
def train(model, data_iterator,optimizer, args):
    model.train()
    loss_avg = utils.RunningAverage()
    t = trange(len(data_iterator), ascii=True)
    for step, _ in enumerate(t):
        # fetch the next training batch
        batch = next(iter(data_iterator))
        batch = tuple(t.to(args.device) for t in batch)
        input_ids, attention_mask, seq_tags, relations, corres_tags, rel_tags = batch

        input_ids = input_ids.to(args.device)
        attention_mask = attention_mask.to(args.device)
        seq_tags = seq_tags.to(args.device)
        relations = relations.to(args.device)
        corres_tags = corres_tags.to(args.device)
        rel_tags = rel_tags.to(args.device)

        loss, loss_seq, loss_mat, loss_rel = model(input_ids,
                                                   attention_mask=attention_mask,
                                                   seq_tags=seq_tags,
                                                   potential_rels=relations,
                                                   corres_tags=corres_tags,
                                                   rel_tags=rel_tags,
                                                   )

        if torch.cuda.device_count() > 1:
            loss = loss.mean()  # mean() to average on multi-gpu.
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            model.zero_grad()

        loss_avg.update(loss.item() * args.gradient_accumulation_steps)
        t.set_postfix(loss='{:05.3f}'.format(loss_avg()))

def evaluate(model, data_iterator, args, mark='Val'):
    # set model to evaluation mode
    model.eval()
    predictions = []
    ground_truths = []
    correct_num, predict_num, gold_num = 0, 0, 0

    for batch in tqdm(data_iterator, unit='Batch', ascii=True):
        # to device
        batch = tuple(t.to(args.device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_ids, attention_mask, triples, input_tokens = batch
        bs, seq_len = input_ids.size()

        # inference
        with torch.no_grad():
            pred_seqs, pre_corres, xi, pred_rels = model(input_ids,
                                                         attention_mask=attention_mask,
                                                         )
            pred_seqs = pred_seqs.detach().cpu().numpy()
            pre_corres = pre_corres.detach().cpu().numpy()
        xi = np.array(xi.cpu())
        pred_rels = pred_rels.detach().cpu().numpy()
        xi_index = np.cumsum(xi).tolist()
        xi_index.insert(0, 0)

        for idx in range(bs):
            pre_triples = tag_mapping_corres(predict_tags=pred_seqs[xi_index[idx]:xi_index[idx + 1]],
                                             pre_corres=pre_corres[idx],
                                             pre_rels=pred_rels[xi_index[idx]:xi_index[idx + 1]],
                                             label2idx_sub=Label2IdxSub,
                                             label2idx_obj=Label2IdxObj)

            gold_triples = span2str(triples[idx], input_tokens[idx])
            pre_triples = span2str(pre_triples, input_tokens[idx])
            ground_truths.append(list(set(gold_triples)))
            predictions.append(list(set(pre_triples)))
            correct_num += len(set(pre_triples) & set(gold_triples))
            predict_num += len(set(pre_triples))
            gold_num += len(set(gold_triples))

    metrics = get_metrics(correct_num, predict_num, gold_num)
    # logging loss, f1 and report
    metrics_str = "; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics.items())
    logging.info("- {} metrics:\n".format(mark) + metrics_str)

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    output = os.path.join(args.model_dir, '{}_output.tsv'.format(mark))

    with open(output, 'a', encoding='utf-8') as f:
        for key in metrics.keys():
            f.write(" {} = {:.4f}\t".format(key, metrics[key]))
        f.write('\n\n')
    return metrics, predictions, ground_truths

def train_and_evaluate(model, args, store_file=None, train_loader=None, val_loader=None, test_loader=None):

    if store_file is not None:
        store_path = os.path.join(args.model_dir, args.store_file + '_model.pth.tar')
        logging.info("Restoring parameters from {}".format(store_path))

        model, optimizer = load_checkpoint(store_path)
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(args.device)
    logger.info("#  Number_of_GPU : {}  #\n".format(torch.cuda.device_count()))

    param_optimizer = list(model.named_parameters())
    # pretrain model param
    param_pre = [(n, p) for n, p in param_optimizer if 'bert' in n]
    # downstream model param
    param_downstream = [(n, p) for n, p in param_optimizer if 'bert' not in n]
    no_decay = ['bias', 'LayerNorm', 'layer_norm']
    optimizer_grouped_parameters = [
        # pretrain model param
        {'params': [p for n, p in param_pre if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay_rate, 'lr': args.fin_tuning_lr
         },
        {'params': [p for n, p in param_pre if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.fin_tuning_lr
         },
        # downstream model
        {'params': [p for n, p in param_downstream if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay_rate, 'lr': args.downs_en_lr
         },
        {'params': [p for n, p in param_downstream if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0, 'lr': args.downs_en_lr
         }
    ]
    num_train_optimization_steps = len(train_loader) // args.gradient_accumulation_steps * args.epoch_num
    optimizer = BertAdam(optimizer_grouped_parameters, warmup=args.warmup_prop, schedule="warmup_cosine",
                         t_total=num_train_optimization_steps, max_grad_norm=args.clip_grad)

    # parallel model
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    best_val_f1 = 0.0
    model.zero_grad()
    optimizer.zero_grad()

    for epoch in range(1, args.epoch_num + 1):
        # Run one epoch
        logger.info("Epoch {}/{}".format(epoch, args.epoch_num))

        # Train for one epoch on training set
        train(model, train_loader, optimizer, args)
        val_metrics, _, _ = evaluate(model, val_loader, args, mark='Val')
        test_metrics, _, _ = evaluate(model, test_loader, args, mark='test')

        val_f1 = val_metrics['f1']
        improve_f1 = val_f1 - best_val_f1

        # Save weights of the network
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        optimizer_to_save = optimizer
        save_checkpoint({'epoch': epoch + 1,
                        'model': model_to_save,
                        'optim': optimizer_to_save},
                        is_best=improve_f1 > 0,
                        checkpoint=args.model_dir)

def save_checkpoint(state, is_best, checkpoint):
    filepath = os.path.join(checkpoint, 'last_model.pth.tar')
    if not os.path.exists(checkpoint):
        logger.info("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.makedirs(checkpoint)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'Optimum_model.pth.tar'))

def load_checkpoint(checkpoint, optimizer=True):

    if not os.path.exists(checkpoint):
        raise ValueError("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))

    if optimizer:
        return checkpoint['model'], checkpoint['optim']
    return checkpoint['model']
