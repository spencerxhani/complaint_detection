import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import argparse
import logging
import torch
import torch.nn.functional as F
from time import strftime, localtime
from config import Configs
from pytorch_pretrained_bert import BertModel
from data_utils import Tokenizer4Bert
from models.bert_ssc import BERT_SSC
from utils import remove_delimiter, remove_separator, remove_empty, remove_two_spaces, remove_three_spaces

# Initialize logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class TextInModel:
    """Load the pre-trained model, you can use your model just as easily.
    """
    def __init__(self, opt):
        self.opt = opt
        if 'bert' in opt.model_name:
            self.tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name, opt.max_num_chunk)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = BERT_SSC(bert, opt).to(opt.device)
            logger.info('loading model {0} ... done'.format(opt.model_name))

            if torch.cuda.is_available():
                # load on gpu environment
                self.model.load_state_dict(torch.load(opt.state_dict_path))
            else:
                # load on environment without gpu
                map_location=torch.device('cpu')
                self.model.load_state_dict(torch.load(opt.state_dict_path, map_location=map_location))            

            # switch model to evaluation mode
            self.model.eval()
            torch.autograd.set_grad_enabled(False)
        else:
            logger.error('Now, we only support bert-based model')
            raise ValueError("Now, we only support bert-based model")

    def text_preprocessing(self, batch_raw_texts):
        try:
            # text-preprocessing
            batch_raw_texts = [remove_delimiter(raw_text) for raw_text in batch_raw_texts]
            batch_raw_texts = [remove_separator(raw_text) for raw_text in batch_raw_texts]
            batch_raw_texts = [remove_empty(raw_text) for raw_text in batch_raw_texts]
            batch_raw_texts = [remove_two_spaces(raw_text) for raw_text in batch_raw_texts]
            batch_raw_texts = [remove_three_spaces(raw_text) for raw_text in batch_raw_texts]
        except Exception as e:
            logger.error("Could not get text preprocessing done : {}".format(e))
            raise
        # the reason we do not loggin succeful information is that will reduce our model latency when go production
        return batch_raw_texts

    def tokenize(self, batch_raw_texts):
        try:
            text_bert_indices = []
            for text in batch_raw_texts:
                ls_tokens = self.tokenizer.text_to_sequence("[CLS] " + text)
                text_bert_indices.append(ls_tokens)
        except Exception as e:
            logger.error("Could not get tokenize done: {}".format(e))
            raise
        return text_bert_indices

    def predict_prob(self, batch_raw_texts):
        """
        batch preprocessing can efficiently bosst qps due to using gpu's nature.
        
        return [batch_size, polarities_dim]
        paras:
            raw_texts: list of string
        """
        try:
            # text-preprocessing
            batch_raw_texts = self.text_preprocessing(batch_raw_texts)
            # tokenize
            text_bert_indices = self.tokenize(batch_raw_texts)
            # conver to tensor
            text_bert_indices = torch.tensor(text_bert_indices, dtype=torch.int64).to(self.opt.device)

            t_inputs = [text_bert_indices]
            t_outputs = self.model(t_inputs)

            t_probs = F.softmax(t_outputs, dim=-1).to(self.opt.device)
        except Exception as e:
            logger.error("Could not predict probability : {}".format(e))
            raise
        return t_probs
    
    def predict(self, batch_raw_texts):
        """
        batch preprocessing can efficiently bosst qps due to using gpu's nature.
        
        return [batch_size, ]
        paras:
            raw_texts: list of string
        """
        try:
            t_probs = self.predict_prob(batch_raw_texts)
            t_preds = torch.argmax(t_probs, dim=1)
        except Exception as e:
            logger.error("Could not predict : {}".format(e))
            raise            
        return t_preds

    def predict_product(self, batch_raw_texts):
        """
        batch preprocessing can efficiently bosst qps due to using gpu's nature.
        
        return array with shape of [batch_size, ] and the element is ["main-product”, “sub-product”]

        paras:
            batch_raw_texts: list of string
        """
        #print (Configs.concatenate_token)
        main_sub_products = [Configs.id_to_main_sub[label].split(Configs.concatenate_token) for label in self.predict(batch_raw_texts).numpy()]
        return main_sub_products

if __name__ == '__main__':
    # Model Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert_ssc', type=str, required = True)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--max_seq_len', default=140, type=int)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--state_dict_path', default="../../artifacts/bert_ssc_val_acc", type=str, help='path to persist model')
    opt = parser.parse_args()
    # loading model
    log_file = '../logs/serve_{}-{}.log'.format(opt.model_name, strftime("%y%m%d-%H%M", localtime()))
    if not os.path.exists('../logs'):
        os.mkdir('../logs')
    logger.addHandler(logging.FileHandler(log_file))

    logger.info("Loading PyTorch model")
    logger.info("Please wait until model has fully loaded")

    model = TextInModel(opt)
    test_case = ["complaint test 1","complaint test 2", "complaint test 3"]
    t_probs = model.predict_prob(test_case)
    print('t_probs = ', t_probs)
    pred = model.predict(test_case)
    print('pred = ', pred)
    main_sub_products = model.predict_product(test_case)
    print('main_sub_products = ', main_sub_products)
