import torch
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel
from torch.utils.data import Dataset
import pandas as pd

class Tokenizer4Bert:
    def __init__(self, max_seq_len, pretrained_bert_name, max_num_chunks):
        # Load pretrained model/tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_bert_name)
        self.max_seq_len = max_seq_len
        self.max_num_chunks = max_num_chunks

    def long_text_to_chunks(self, text):
        """return an array with shape of [30, max_seq_len], and the element is the token representation of the BERT"""
        import numpy as np
        ls_of_tokens = self.tokenizer.tokenize(text)
        #print('ls of tokens: {},len:{}'.format(ls_of_tokens, len(ls_of_tokens)))
        n = len(ls_of_tokens) // self.max_seq_len
        #print('n: {} '.format(n))
        res = []
        for i in range(self.max_num_chunks):
            if i < n:
                sub_ls_of_tokens = ls_of_tokens[i*self.max_seq_len:i*self.max_seq_len + self.max_seq_len]
            elif i == n:
                tmp_len = len(ls_of_tokens[i*self.max_seq_len:])
                sub_ls_of_tokens = ls_of_tokens[i*self.max_seq_len:]+['[PAD]']*(self.max_seq_len-tmp_len)
            else:
                sub_ls_of_tokens = ['[PAD]']*self.max_seq_len
            # convert ls of toens to sequence of ids
            sub_ls_of_tokens = self.tokenizer.convert_tokens_to_ids(sub_ls_of_tokens)
            res.append(sub_ls_of_tokens)
        return np.array(res)

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        sequence = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(text))
        print("seq: {}, len:{}".format(sequence,len(sequence)))
        if len(sequence) == 0:
            sequence = [0]
            print('seq:{}'.format(sequence))
        if reverse:
            sequence = sequence[::-1]
            print('seq:{}'.format(sequence))

        return self.pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

    def pad_and_truncate(self, sequence, maxlen, dtype='int64', padding='post', truncating='post', value=0):
        import numpy as np
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class ABSADataset(Dataset):
    def __init__(self, fname, tokenizer):
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()[:100]
        fin.close()

        all_data = []
        for i in range(0, len(lines), 2):
            text = lines[i].strip()
            polarity = lines[i + 1].strip()
            # single-sentence classification
            text_raw_bert_indices = tokenizer.text_to_sequence("[CLS] " + text)
            # documentation classification
            text_raw_bert_documents = tokenizer.long_text_to_chunks("[CLS] " + text)
            # label
            polarity = int(polarity)  # range betwee 0 to num_class -1

            data = {
                'text_raw_bert_indices': text_raw_bert_indices,
                'text_raw_bert_documents': text_raw_bert_documents,
                'polarity': polarity,
            }
            all_data.append(data)

        self.data = all_data

    def get_dataframe(self, tokenizer):
        """
        Conver dataset into DataFrame(Pandas)
        It's only support for bert-based model.
        """
        df = []
        columns_name = []
        for i in range(len(self.data)):
            tmp = []
            for k, v in self.data[i].items():
                try:
                    to_str = " ".join(tokenizer.tokenizer.convert_ids_to_tokens(v))
                    tmp.append(to_str)
                except:
                    if k == 'aspect_in_text':
                        # it's a 1-D tensor wtih shape of (2,), representing the start and end index of the aspect
                        v = v.numpy()  # 1-D tensor
                        #print (v.shape)
                    tmp.append(v)
                if i <= 0:
                    columns_name.append(k)
            df.append(tmp)
        df = pd.DataFrame(df,columns=columns_name)
        return df

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
