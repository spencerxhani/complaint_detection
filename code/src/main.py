"""
python3 main.py --train True
python3 main.py --predict "My father attempted to clear a debt with a collection agency. They informed him that he could settle the debt for a specific amount. He received a letter stating that if wanted to resolve the issue all he would have to do would pay the settlement amount."
"""
import argparse
from predict import TextInModel
from data_process import DataProcess
from train import Instructor

def main():
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', default = False, type = bool, help = "if True, pulls the raw data from the DB and generates the fitted model artifact. Otherwise, ignore")
    parser.add_argument('--predict', default='[test sentence]', type=str, help = "a complaint text that u want to extract their main an sub products")

    # Parameter for data preprocessing
    parser.add_argument('--data_path', default='../../data', type=str, help = "data path")
    parser.add_argument('--output_path', default='../../data', type=str, help = "path to save train/test text data ")
    parser.add_argument('--train_test_ratio', default=0.2, type=float, help='set ratio between 0 and 1 for train/test split')
    parser.add_argument('--text_column', default="COMPLAINT_TEXT", type=str, help='text column')
    parser.add_argument('--label_column', default="class_label", type=str, help='text column')
    parser.add_argument('--N', default=50, type=int, help='N is most frequent N sub-products that our model will cover.')
    parser.add_argument('--seed', default=None, type=int, help='set seed for reproducibility and experimentation')
    parser.add_argument('--load_from_db', default=1, type=int, help='load data from db or local file')
    parser.add_argument('--tb_list', default=['complaints_users', 'complaints_companies', 'issues', 'products'], type=list, help='list of table names you want to query from db')

    # Model Parameters for train
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=2e-5, type=float, help='try 5e-5, 2e-5 for BERT, 1e-3 for others')
    parser.add_argument('--l2reg', default=0.01, type=float)
    parser.add_argument('--num_epoch', default=10, type=int, help='try larger number for non-BERT models')
    parser.add_argument('--max_seq_len', default=150, type=int)
    parser.add_argument('--max_num_chunk', default=3, type=int)
    parser.add_argument('--batch_size', default=64, type=int, help='try 16, 32, 64 for BERT models')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--valset_ratio', default=0, type=float, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--get_tokenized_result', default=False, type=bool, help='to see if save the tokenized result')

    # Model Parameters for predict
    parser.add_argument('--model_name', default='bert_ssc', type=str)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--bert_dim', default=768, type=int)
    parser.add_argument('--polarities_dim', default=51, type=int)
    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0')
    parser.add_argument('--state_dict_path', default="../../artifacts/bert_ssc_val_acc", type=str, help='path to persist model')

    opt = parser.parse_args()
    # train
    if opt.train == True:
        # load data from db and manipulate into our model training/validating format
        dp = DataProcess(opt)
        dp.run()
        # train
        ins = Instructor(opt)
        ins.run()
    # predict
    if opt.predict == "[test sentence]":
        pass
    else:
        model = TextInModel(opt)
        text = [opt.predict]
        main_sub_products = model.predict_product(text)
        if main_sub_products == [['Others Sub Product']]:
            print ('*' * 100)
            print ("prediction")
            print ('*' * 100)
            print('main product = {}, sub product = {}'.format('Others Main Product', 'Others Sub Product'))
        else:
            print ('*' * 100)
            print ("prediction")
            print ('*' * 100)
            print('main product = {}, sub product = {}'.format(main_sub_products[0][0], main_sub_products[0][1]))

if __name__ == "__main__":
    main()