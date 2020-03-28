# -*- coding: utf-8 -*-
import logging
import argparse
import math
import os
from time import strftime, localtime
import random
import numpy
from pytorch_pretrained_bert import BertModel
from sklearn.metrics import precision_score, recall_score, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from data_utils import Tokenizer4Bert, ABSADataset
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.bert_ssc import BERT_SSC
from models.bert_document_lstm import BERT_DocumentLSTM

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logging.getLogger("pytorch_pretrained_bert.tokenization").setLevel(logging.ERROR)


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        if opt.seed is not None:
            random.seed(opt.seed)
            numpy.random.seed(opt.seed)
            torch.manual_seed(opt.seed)
            torch.cuda.manual_seed(opt.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        model_classes = {
            'bert_ssc':BERT_SSC,
            'bert_document_lstm':BERT_DocumentLSTM
        }
        dataset_files = {
            "train":os.path.join(opt.data_path, "train.txt"),
            "test":os.path.join(opt.data_path, "test.txt")
        }
        input_colses = {
            'bert_ssc': ['text_raw_bert_indices'],
            'bert_document_lstm': ['text_raw_bert_documents']
        }
        initializers = {
            'xavier_uniform_': torch.nn.init.xavier_uniform_,
            'xavier_normal_': torch.nn.init.xavier_normal,
            'orthogonal_': torch.nn.init.orthogonal_,
        }
        optimizers = {
            'adadelta': torch.optim.Adadelta,  # default lr=1.0
            'adagrad': torch.optim.Adagrad,  # default lr=0.01
            'adam': torch.optim.Adam,  # default lr=0.001
            'adamax': torch.optim.Adamax,  # default lr=0.002
            'asgd': torch.optim.ASGD,  # default lr=0.01
            'rmsprop': torch.optim.RMSprop,  # default lr=0.01
            'sgd': torch.optim.SGD,
        }
        opt.model_class = model_classes[opt.model_name]
        opt.dataset_file = dataset_files
        opt.inputs_cols = input_colses[opt.model_name]
        opt.initializer = initializers[opt.initializer]
        opt.optimizer = optimizers[opt.optimizer]
        opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
            if opt.device is None else torch.device(opt.device)

        if 'bert' in opt.model_name:
            tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name, opt.max_num_chunk)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], tokenizer)
        self.testset = ABSADataset(opt.dataset_file['test'], tokenizer)

        if opt.get_tokenized_result == True:
            try:
                df_tokenize_train = self.trainset.get_dataframe(tokenizer)
                df_tokenize_train.to_csv("../../data/df_tokenize_train.csv", index = False)
            except Exception as e:
                logger.error("Could not get tokenizer dataframe : {}".format(e))
                raise
            logger.info("save tokenizer train df successfully")

        assert 0 <= opt.valset_ratio < 1, logger.error("Please check if validation ratio is between 0 and 1")

        if opt.valset_ratio > 0:
            valset_len = int(len(self.trainset) * opt.valset_ratio)
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)-valset_len, valset_len))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
                    if p.requires_grad:
                        if len(p.shape) > 1:
                            self.opt.initializer(p)
                        else:
                            stdv = 1. / math.sqrt(p.shape[0])
                            torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, val_data_loader):
        max_val_acc = 0
        max_val_f1 = 0
        global_step = 0
        path = None
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 100)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total, loss_total = 0, 0, 0
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                # sample_batched: dict of tensor with shape of [Batch size, Max seq length]
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()
                # list of tensor with shape of [Batch size, Max seq length]
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                # [Batch size, Max Num Chunks, Max seq length]
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_total += len(outputs)
                loss_total += loss.item() * len(outputs)
                if global_step % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    logger.info('loss: {:.4f}, acc: {:.4f}'.format(train_loss, train_acc))

            val_acc, val_f1, val_p, val_r = self._evaluate_acc_f1(val_data_loader)
            logger.info('> val_acc: {:.4f}, val_f1: {:.4f}, val_p: {:.4f}, val_r: {:.4f}'.format(val_acc, val_f1, val_p, val_r))
            if val_acc > max_val_acc:
                max_val_acc = val_acc
                if not os.path.exists('../../artifacts'):
                    os.mkdir('../../artifacts')
                path = '../../artifacts/{0}_val_acc'.format(self.opt.model_name)
                torch.save(self.model.state_dict(), path)
                logger.info('>> saved: {}'.format(path))
            if val_f1 > max_val_f1:
                max_val_f1 = val_f1

        return path

    def _evaluate_acc_f1(self, data_loader):
        n_correct, n_total = 0, 0
        t_targets_all, t_outputs_all = None, None
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(self.opt.device)
                t_outputs = self.model(t_inputs)

                n_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                n_total += len(t_outputs)

                if t_targets_all is None:
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)

        acc = n_correct / n_total
        f1 = f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[i for i in range(self.opt.polarities_dim)], average='macro')
        precision = precision_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[i for i in range(self.opt.polarities_dim)], average='macro')
        recall = recall_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[i for i in range(self.opt.polarities_dim)], average='macro')
        return acc, f1, precision, recall

    def run(self):
        # Loss and Optimizer
        criterion = nn.CrossEntropyLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=self.opt.batch_size, shuffle=False)
        val_data_loader = DataLoader(dataset=self.valset, batch_size=self.opt.batch_size, shuffle=False)

        self._reset_params()
        best_model_path = self._train(criterion, optimizer, train_data_loader, val_data_loader)
        self.model.load_state_dict(torch.load(best_model_path))
        self.model.eval()
        test_acc, test_f1, test_p, test_r = self._evaluate_acc_f1(test_data_loader)
        logger.info('>> test_acc: {:.4f}, test_f1: {:.4f}, test_precision: {:.4f}, test_recall: {:.4f}'.format(test_acc, test_f1, test_p, test_r))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_ssc', type=str, required = True)
    parser.add_argument('--data_path', default='../../data', type=str, help = "data path")
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--max_seq_len', default=100, type=int)
    parser.add_argument('--max_num_chunk', default=3, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility')
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--get_tokenized_result', default=False, type=bool, help='to see if save the tokenized result')

    opt = parser.parse_args()

    # logging
    log_file = '../logs/train_{}-{}.log'.format(opt.model_name, strftime("%y%m%d-%H%M", localtime()))
    if not os.path.exists('../logs'):
        os.mkdir('../logs')
    logger.addHandler(logging.FileHandler(log_file))

    ins = Instructor(opt)
    ins.run()

if __name__ == '__main__':
    main()
