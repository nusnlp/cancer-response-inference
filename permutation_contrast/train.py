import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from argparse import ArgumentParser
import json
import re
from tqdm import tqdm, trange
from pprint import pprint, pformat
import time
from datetime import timedelta
from shutil import copyfile

import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim import Adam
import random
from model import Identity
from nemo.collections.nlp.models.language_modeling.megatron_bert_model import MegatronBertModel
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import classification_report, confusion_matrix

from transformers import AdamW

from utils import soft_cross_entropy

def fseed(seed=62):
    pl.seed_everything(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def set_yaml_model_seed(seed, dsave):
    import yaml
    with open(dsave) as f:
        doc = yaml.safe_load(f)
    doc['seed'] = seed
    doc['precision'] = 32
    with open(dsave, 'w') as f:
        yaml.dump(doc, f)

def read_json(d):
	with open(d, 'r', encoding='utf-8') as f:
		return json.load(f)

def write_json(data, dsave):
	outf = open(dsave, 'w', encoding='utf-8')
	json.dump(data, outf, indent=2, ensure_ascii=False)
	outf.close()
	print('>>> write to {}'.format(dsave))

def write_jsonl(data, dsave):
	outf = open(dsave, 'w')
	for d in data:
		json.dump(d, outf)
		outf.write('\n')
	outf.close()
	print('\n+++ write to {}'.format(dsave))

def custom_tokenize(text, tokenizer):
    encoded = tokenizer.encode_plus(text,
                                    add_special_tokens=True,
                                    return_token_type_ids=True)
    return encoded['input_ids']

def summary_printout(logger, content):
    for k, v in content.items():
        logger.critical('-- {}: {}'.format(k, v))
    logger.critical('\n------------------------------\n')

def format_numbers(n, decimal=4):
    d = decimal
    ns = '{:.{}f}'.format(n, d)
    return float(ns)

def format_time(seconds):
    return str(timedelta(seconds=seconds)).split('.')[0]

def save_configs(dsave, task_config):
    write_json(task_config, dsave + '/config.json')
    if os.path.exists('utils.py'):
        copyfile('utils.py', dsave + '/utils.py')
    copyfile('model.py', dsave + '/model.py')
    copyfile('train.py', dsave + '/train.py')

class Trainer():
    def __init__(self, args):
        self.args = args
        self.task_config = read_json(args.config)
        ckpt_dsave = './saved/perm_contrast_nemo_{}'.format('gatortron')
        assert not os.path.exists(ckpt_dsave)
        os.makedirs(ckpt_dsave)
        self.ckpt_dsave = ckpt_dsave
        print('{} created'.format(ckpt_dsave))
        self.dev_dsave = ckpt_dsave + '/dev_pred.jsonl'
        self.test_dsave = ckpt_dsave + '/test_pred.jsonl'
        self.device = torch.device('cuda:{}'.format(0))
        self.task_config['train_data_path'] = '/path/to/permuted_data/trainS_x11.json'

        save_configs(ckpt_dsave, self.task_config)

        self.test_data_file = read_json(self.task_config['test_data_path'])
        self.LABELS = self.test_data_file['labels']
        self.n_class = len(self.LABELS)

        self.N_copies = 11

        self.model_path = 'ckpt/GatorTron-OG_nemo/MegatronBERT.nemo'

        nemo_config = OmegaConf.load(args.nemo_config)
        print('precision: ', nemo_config.trainer.precision)
        self.trainer = pl.Trainer(**nemo_config.trainer)
        override_config_path = '/abs/path/to/GatorTron-OG_nemo/model_config.yaml'   # absolute path is required
        set_yaml_model_seed(self.args.sd, override_config_path)
        self.lmodel = MegatronBertModel.restore_from(self.model_path, override_config_path=override_config_path, trainer=self.trainer)

        self.lmodel_lm_head = self.keep_state_dict(self.lmodel.model.lm_head)
        self.lmodel_binary_head = self.keep_state_dict(self.lmodel.model.binary_head)
        self.lmodel_language_model_pooler = self.keep_state_dict(self.lmodel.model.language_model.pooler)
        self.lmodel.model.lm_head = Identity()
        self.lmodel.model.binary_head = Identity()
        self.lmodel.model.language_model.pooler = Identity()

        emb_dim = self.lmodel.cfg.hidden_size
        self.layer_cls = nn.Linear(emb_dim, self.n_class)

        self.mse_loss = torch.nn.MSELoss()
        self.kldiv_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.vsoftmax = torch.nn.Softmax(dim=1)

        bsz = 8

        self.optimizer = None

        bsz_test = 16

        tokenizer = self.lmodel.tokenizer
        self.tokenizer = tokenizer
        self.cls_token_id = tokenizer.cls_id
        self.pad_token_id = tokenizer.pad_id
        self.sep_token_id = tokenizer.sep_id

        train_data = self.load_data(read_json(self.task_config['train_data_path'])['data'],
                                                 tokenizer, shuffle=False)
        dev_data = self.load_data(read_json(self.task_config['dev_data_path'])['data'], tokenizer)
        test_data = self.load_data(self.test_data_file['data'], tokenizer)

        if args.debug:
            train_data = train_data[-100:]
            dev_data = dev_data[-100:]
            test_data = test_data[-100:]
            self.epochs = 5

        self.N_train = int(len(train_data)/self.N_copies)
        train_data_contrast = train_data[-self.N_train:] + train_data[:-self.N_train]

        self.train_data = [train_data[i:i + bsz] + train_data_contrast[i:i + bsz] for i in range(0, len(train_data), bsz)]
        self.dev_data = [dev_data[i:i + bsz_test] for i in range(0, len(dev_data), bsz_test)]
        self.test_data = [test_data[i:i + bsz_test] for i in range(0, len(test_data), bsz_test)]

    def load_model(self, model_path):
        self.loaded = torch.load(model_path, map_location='cuda:{}'.format(self.args.cuda))
        self.lmodel.load_state_dict(self.loaded['lmodel'])
        self.layer_cls.load_state_dict(self.loaded['layer_cls'])

    def load_data(self, data, tokenizer, shuffle=False):
        if shuffle:
            data = random.sample(data, len(data))
        datap = []

        token_len_list = []

        for inst in data:
            uid = inst['sn_report_number'] + ' ||| ' + inst['report_date']
            if 'label_pred' in inst:
                label = inst['label_pred']
            else:
                label = inst['label']
            label_probs = [0] * self.n_class
            label_probs[label] = 1

            if 'input_ids' in inst:
                combined_tokens = inst['input_ids']
            else:
                ctext = inst['conclusion']

                combined_text = ctext

                combined_tokens = self.tokenizer.text_to_ids(self.tokenizer.cls_token + combined_text + self.tokenizer.sep_token)
                token_len_list.append(len(combined_tokens))

                max_seq_len = 300
                if len(combined_tokens) > max_seq_len:
                    combined_tokens = combined_tokens[:max_seq_len]

            metadata = {}
            metadata['uid'] = uid

            datap_inst = {'combined_tokens': combined_tokens,
                          'label': label,
                          'label_probs': label_probs,
                          'metadata': metadata
                          }
            datap.append(datap_inst)

        return datap

    def output_test_results(self, test_data, test=False):
        test_labels = [d['label'] for d in test_data]
        test_preds = [d['label_pred'] for d in test_data]
        report = classification_report(test_labels, test_preds, digits=4,
                                       labels=[i for i in range(self.n_class)], target_names=self.LABELS, output_dict=True)

        if 'accuracy' in report.keys():
            acc = report['accuracy']
        else:
            self.logger.critical('!!! micro avg !!!')
            acc = report['micro avg']['f1-score']
        macro_f1 = report['macro avg']['f1-score']
        wavg_f1 = report['weighted avg']['f1-score']

        if test:
            report2 = classification_report(test_labels, test_preds, digits=4,
                                           labels=[i for i in range(self.n_class)], target_names=self.LABELS, output_dict=False)
            self.logger.critical('#'*56)
            self.logger.critical(report2)
            self.logger.critical('#'*56)

        return acc, macro_f1, wavg_f1

    def log_test_results(self, test_data, test_name, test=False):
        acc, macro_f1, wavg_f1 = self.output_test_results(test_data, test=test)
        self.logger.critical('{:<9}  {:.2f} / {:.2f} / {:.2f}'.
                             format(test_name, acc * 100, macro_f1 * 100, wavg_f1 * 100))
        return acc, macro_f1, wavg_f1

    def run_process(self):
        global_start_time = time.time()
        ckpt_dsave = self.ckpt_dsave
        logger = logging.getLogger(self.__class__.__name__)
        logger.setLevel(logging.DEBUG)
        fh = logging.FileHandler(os.path.join(ckpt_dsave, 'train.log'))
        fh.setLevel(logging.CRITICAL)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setLevel(logging.CRITICAL)
        logger.addHandler(ch)
        self.logger = logger
        logger.critical('------ task config ------')
        logger.critical('-- cuda: {}'.format(self.args.cuda))
        summary_printout(logger, self.task_config)

        self.lmodel = self.lmodel.to(self.device)
        self.layer_cls = self.layer_cls.to(self.device)

        param_groups = [{'params': self.lmodel.parameters()},
                        {'params': self.layer_cls.parameters()}]

        lr = 2e-5

        epochs = 50
        patience_max = 2

        self.optimizer = Adam(params=param_groups, lr=lr, betas=(0.9,0.999), weight_decay=0.01)

        ckpt_best = None
        dev_acc_best = 0.0
        epoch_best = 0
        patience = 0

        summary = []

        epoch_i = 0

        for epoch in trange(epochs, desc='epoch', ncols=100):
            epoch_start_time = time.time()

            if epoch_i == 0:
                test_out, _ = self.run_subprocess(self.test_data, test=True)
                self.log_test_results(test_out, 'Test (t=0)\n', test=True)


            _, train_acc, loss_terms = self.run_subprocess(self.train_data, train=True)
            dev_out, dev_acc, valid_loss_terms = self.run_subprocess(self.dev_data, dev=True)
            dev_acc, _, dev_wavg_f1 = self.log_test_results(dev_out, 'Dev')

            epoch_i += 1

            test_out, test_acc = self.run_subprocess(self.test_data, test=True)
            test_acc, _, test_wavg_f1 = self.log_test_results(test_out, 'Test', test=True)

            if dev_acc > dev_acc_best:
                dev_acc_best = dev_acc
                epoch_best = epoch
                ckpt_best = {'epoch': epoch_best,
                             # 'lmodel': self.keep_state_dict(self.lmodel.cpu().state_dict()),
                             'layer_cls': self.keep_state_dict(self.layer_cls.cpu().state_dict()),
                             'train_acc': train_acc,
                             'dev_acc': dev_acc_best,
                             'test_acc': test_acc
                             }
                self.lmodel = self.lmodel.to(self.device)
                self.layer_cls = self.layer_cls.to(self.device)

                write_jsonl(dev_out, self.dev_dsave)
                write_jsonl(test_out, self.test_dsave)

                patience = 0
            else:
                patience += 1

            summary.append({'epoch': epoch,
                            'epoch_best': epoch_best,
                            'train_acc': train_acc,
                            'dev_acc': dev_acc,
                            'test_acc': test_acc,
                            'epoch_time': format_time(time.time() - epoch_start_time)
                            })
            logger.critical('\n\n------ summary: epoch {} ----\n'.format(epoch))
            logger.critical('-- train loss ---> {}'.format(loss_terms))
            logger.critical('-- valid loss ---> {}'.format(valid_loss_terms))
            summary_printout(logger, summary[-1])

            if patience == patience_max or epoch == epochs - 1:
                torch.save(ckpt_best, ckpt_dsave + '/linear.pt.tar')
                self.lmodel.model.lm_head = self.lmodel_lm_head
                self.lmodel.model.binary_head = self.lmodel_binary_head
                self.lmodel.model.language_model.pooler = self.lmodel_language_model_pooler
                self.lmodel.save_to(ckpt_dsave + '/model.nemo')
                self.lmodel.model.lm_head = Identity()
                self.lmodel.model.binary_head = Identity()
                self.lmodel.model.language_model.pooler = Identity()
                if patience == patience_max:
                    logger.critical('best epoch: {}. patience {} reached.'.format(epoch_best, patience_max))
                else:
                    logger.critical('best epoch: {}.'.format(epoch_best))
                logger.critical('------ training summary ------')
                summary_printout(logger, summary[epoch_best])
                logger.critical('total epochs: {}'.format(epoch))
                logger.critical('total time: {}'.format(format_time(time.time() - global_start_time)))
                logger.critical('model directory: {}'.format(ckpt_dsave))
                logger.critical('------------------------------')
                logger.critical('best ckpt saved: model.pt.tar')

                break

    def keep_state_dict(self, state_dict):
        import copy
        return copy.deepcopy(state_dict)

    def run_subprocess(self, data, train=False, dev=False, test=False, gen=False, unlabeled=False):
        if train or unlabeled:
            self.lmodel.train()
            self.layer_cls.train()
        else:
            self.lmodel.eval()
            self.layer_cls.eval()

        device = self.device
        train_loss = 0.0
        valid_loss_CE = 0.0
        run_correct = 0
        seen = 0
        inst_count = 0
        loss_terms = {}

        sum_CE = 0.0
        sum_contrast = 0.0

        acc = 0.0
        # train
        # batch_gen = tqdm(data)
        pred_out = []

        val_iteration = int(len(data)/self.N_copies)
        # val_iteration = len(data)
        if train == False:
            val_iteration = len(data)
            self.dataiter = iter(data)

        iters = trange(val_iteration, ncols=100)
        # for batch in batch_gen:
        for batch_idx in iters:
            if train:
                try:
                    batch = next(self.trainiter)
                except:
                    self.trainiter = iter(data)
                    batch = next(self.trainiter)
            else:
                batch = next(self.dataiter)

            if train or unlabeled:
                self.optimizer.zero_grad()
            combined_tokens_b = pad_sequence([torch.LongTensor(d['combined_tokens']) for d in batch], batch_first=True,
                                             padding_value=self.pad_token_id).to(device)
            attn_mask_b = pad_sequence([torch.FloatTensor([1]*len(d['combined_tokens'])) for d in batch], batch_first=True,
                                        padding_value=0).to(device)
            token_type_ids = torch.full(size=attn_mask_b.size(), fill_value=0, dtype=torch.long, device=device)

            if not (unlabeled or gen):
                labels = torch.LongTensor([d['label'] for d in batch]).to(device)
                label_probs = torch.tensor([d['label_probs'] for d in batch]).to(device)

                label_gold = [d['label'] for d in batch]

            if not train:
                with torch.no_grad():
                    model_out = self.lmodel(input_ids=combined_tokens_b, attention_mask=attn_mask_b,
                                            token_type_ids=token_type_ids)
                    combined_out = model_out[0][:, 0, :]
                    x = self.layer_cls(combined_out)
                    if gen:
                        gen_tau = 1.0
                        output = self.vsoftmax(torch.div(x, gen_tau))
                    else:
                        output = self.vsoftmax(x)
                    _, pred_label = output.max(1)  # B

            if train:
                model_out = self.lmodel(input_ids=combined_tokens_b, attention_mask=attn_mask_b,
                                        token_type_ids=token_type_ids)
                combined_out = model_out[0][:, 0, :]
                x = self.layer_cls(combined_out)
                output = self.vsoftmax(x)
                output_a = output[:int(output.shape[0]/2)]
                output_b = output[int(output.shape[0]/2):]

            if unlabeled:
                pass


            seen += len(batch)
            if train:
                # p1 = 0.8
                p2 = 10.0
                # p2 = 1.0
                # L_CE = self.cross_entropy(x, labels)
                L_CE = soft_cross_entropy(x, label_probs)
                L_contrast = self.mse_loss(output_a, output_b)

                sum_CE += L_CE.item() * len(batch)
                sum_contrast += L_contrast.item() * len(batch)

                avg_CE = sum_CE / seen
                avg_contrast = sum_contrast / seen

                # loss = L_CE
                loss = L_CE + p2*L_contrast
                loss.backward()
                train_loss += (loss.item() * len(batch))      # criterion default reduction = 'mean'
                avg_loss = train_loss / seen
                self.optimizer.step()

                loss_terms = {'CE': format_numbers(avg_CE),
                              'cont': format_numbers(avg_contrast),
                              'AggAvg': format_numbers(avg_loss)}

            if unlabeled:
                pass

            if dev:
                with torch.no_grad():
                    L_CE = self.cross_entropy(x, labels)
                    # batch_loss = L_CE

                    valid_loss_CE += L_CE.item() * len(batch)

                    valid_loss_CE_avg = valid_loss_CE / seen

                    valid_avg_loss = valid_loss_CE_avg


                    loss_terms = {'CE': format_numbers(valid_loss_CE_avg),
                                  'sum': format_numbers(valid_avg_loss)}


            if not (unlabeled or gen):
                run_correct += (output.detach().cpu().argmax(dim=1) == labels.cpu()).sum().item()
                # acc = 0.0
                acc = run_correct/seen


            # update tqdm bar display

            if train:
                # batch_gen.set_description('loss: {:.4f} | acc_cls:{:.4f}'.format(avg_loss, acc), refresh=False)
                iters.set_description('CE:{:.3f}|con:{:.3f}|Acc:{:.4f}'.
                                          format(avg_CE, avg_contrast, acc), refresh=False)
            if dev:
                iters.set_description('[dev] acc_cls:{:.4f}'.format(acc), refresh=False)
            if test:
                iters.set_description('[test] acc_cls:{:.4f}'.format(acc), refresh=False)


            if dev or test:
                assert len(batch) == output.shape[0] == pred_label.shape[0] == \
                       x.shape[0] == len(label_gold)
                for d, logits, dist, cls, lgold_hard in zip(batch, x, output, pred_label, label_gold):
                    pred_out.append({'uid': d['metadata']['uid'],
                                     'label': lgold_hard,
                                     'logits': logits.tolist(),
                                     'label_probs': dist.tolist(),
                                     'label_pred': int(cls)
                                     })

                    inst_count += 1


        if train or dev or unlabeled:
            return pred_out, acc, loss_terms
        else:
            return pred_out, acc

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('--config', default='./config.json')
    parser.add_argument('--nemo_config', default='./nemo_files/nemo_lm_config.yaml')
    parser.add_argument('--cuda', default=0, type=int)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--sd', default=200, type=int)
    args = parser.parse_args()

    print('use device: {}'.format(os.environ["CUDA_VISIBLE_DEVICES"]))
    fseed(seed=args.sd)
    trainer = Trainer(args)
    trainer.run_process()
