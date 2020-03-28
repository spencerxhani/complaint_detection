# -*- coding: utf-8 -*-
# file: bert_document_lstm.py
# author: yunruili <ray811030@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from torch.nn import LSTM


class BERT_DocumentLSTM(nn.Module):
    """Long Documentation classification"""
    def __init__(self, bert, opt):
        super(BERT_DocumentLSTM, self).__init__()
        self.opt = opt
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.lstm = LSTM(input_size = opt.bert_dim, hidden_size = opt.bert_dim, batch_first = True) # [B, max_num_chunks, bert_dim]
        self.dense = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        text_bert_indices = inputs[0] # [B, max_num_chunks, max_seq_length]
        # extract bert representation
        lstm_input = [] # [B, max_num_chunk, bert_dim]
        B = text_bert_indices.size()[0]
        for i in range(B):
            document = text_bert_indices[i,:,:]
            _, pooled_output = self.bert(document, output_all_encoded_layers=False) 
            # [max_num_chunk, opt.bert_dim]
            pooled_output = self.dropout(pooled_output)
            lstm_input.append(pooled_output)
        lstm_input = torch.stack(lstm_input)
        # lstm
        _, (h_n, _) = self.lstm(lstm_input) 
        # classifier
        logits = self.dense(h_n[0]) # [B, bert_dim]
        return logits