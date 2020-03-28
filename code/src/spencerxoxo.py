import torch
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel
from torch.utils.data import Dataset
import pandas as pd

from data_utils import Tokenizer4Bert, ABSADataset


# path = '/Users/Spencer/Desktop/nlp_task/complaint_detection/data/train.txt'

# trainset = ABSADataset(path, tokenizer)
# df = trainset.get_dataframe(tokenizer)
#print(df.shape, df.head(1),df.columns,df['text_raw_bert_indices'],df['text_raw_bert_documents'], df['polarity'])
text = 'In this post, I want to show how to apply BERT to a simple text classification problem. I assume that you’re more or less familiar with what BERT is on a high level, and focus more on the practical side by showing you how to utilize it in your work. Roughly speaking, BERT is a model that knows to represent text. You give it some sequence as an input, it then looks left and right several times and produces a vector representation for each word as the output. In their paper, the authors describe two ways to work with BERT, one as with “feature extraction” mechanism. That is, we use the final output of BERT as an input to another model.'
# print(len(text))
# # Tokenizer4Bert(10, 'bert-base-uncased', 3).long_text_to_chunks(text)
# tokenizer = Tokenizer4Bert(10, 'bert-base-uncased', 3)
# tokenizer.text_to_sequence(text)

data = [{
    'text_raw_bert_indices': [1,1,1,1],
    'text_raw_bert_documents': [1,1,1,1222],
    'polarity': 0,
}, {'text_raw_bert_indices': [2,1,1,1],
'text_raw_bert_documents': [2,1,1,1222],
'polarity': 1}]
print(data[0])
k, v, c = data[0].items()
print(k,v,c)
