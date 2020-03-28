import os
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from utils import remove_delimiter, remove_separator, remove_empty, remove_two_spaces, remove_three_spaces, df_to_txt
import pickle
from time import strftime, localtime
import random
import numpy
import logging
import sys
import yaml

# database argu
CONFIG_PATH = "../../misc/db_config.yaml"
SCHEMA_PATH = "../../misc/schemas.yaml"
# logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

class DBConnection:

    def __init__(self, db_config_file):
        with open(db_config_file) as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        self.db_config = config.get("pg")

    def __enter__(self):
        import psycopg2 as pg
        logging.info("Creating DB connection...")
        self.connection = pg.connect(
            host=self.db_config.get("host"),
            port=int(self.db_config.get("port")),
            dbname=self.db_config.get("dbname"),
            user=self.db_config.get("user")
        )
        logging.info("Connection created!")
        return self.connection

    def __exit__(self, type, value, traceback):
        logging.info("Closing the DB connection!")
        self.connection.close()

class RetrieveDB:
    def __init__(self, db_config_path, schema_config):
        self.db_config_path = db_config_path
        with open(schema_config) as schema_file:
            self.schema = yaml.load(schema_file, Loader=yaml.FullLoader)
        self.schema_hash_table = {content_dict['name']:[str.upper(i) for i in content_dict['columns']] for content_dict in self.schema}

    def get_values(self, table_name: str):
        with DBConnection(self.db_config_path) as connection:
            try:
                c = connection.cursor()
                c.execute(f"SELECT * FROM {table_name};")
                result = c.fetchall()
            except Exception as e:
                logger.error("Could not get value from table {}:".format(table_name), e)
                raise
            logger.info("get values from the DB successfully")
        return result
    
    def get_dataframe(self, table_name):
        try:
            df = pd.DataFrame(self.get_values(table_name), columns = self.schema_hash_table[table_name])
        except Exception as e:
            logger.error("Could not confert table {} into pandas DataFrame object:".format(table_name), e)
            raise
        logger.info("confert table into pandas DataFrame successfully")
        return df

    def run(self, tb_list):
        try:
            tb_dict = {}
            for tb_name in tb_list:
                tb_dict[tb_name] = self.get_dataframe(tb_name)
        except Exception as e:
            logger.error("Could not get required pandas DataFrame object I want:", e)
            raise
        logger.info("Get required pandas DataFrame successfully")
        return tb_dict

class DataProcess:
    def __init__(self, opt):
        self.opt = opt
        self.concatenate_token = "-[CON]-"
        self.tb_list = opt.tb_list

    def load_data(self):
        try:
            # load data
            self.complaints_users = pd.read_csv(os.path.join(self.opt.data_path,"complaints_users.csv"))
            self.complaints_companies = pd.read_csv(os.path.join(self.opt.data_path,"complaints_companies.csv"))
            self.issues = pd.read_csv(os.path.join(self.opt.data_path,"issues.csv"))
            self.products = pd.read_csv(os.path.join(self.opt.data_path,"products.csv"))
        except Exception as e:
            logger.error("Could not load data : {}".format(e))
            raise
        logger.info("loading data successfully")

    def retrieve_data_from_db(self):
        try:
            # retrive data
            tb_dict = RetrieveDB(CONFIG_PATH, SCHEMA_PATH).run(self.tb_list)
            self.complaints_users = tb_dict['complaints_users']
            self.complaints_companies = tb_dict['complaints_companies']
            self.issues = tb_dict['issues']
            self.products = tb_dict['products']
        except Exception as e:
            logger.error("Could not retrieve data : {}".format(e))
            raise
        logger.info("retrieving data successfully")

    def data_processing(self):
        try:
            # data porcessing
            self.complaints_users.rename(columns = {"DATE":"DATE_USER_SUBMITTED"}, inplace = True)
            self.complaints_companies.rename(columns = {"DATE":"DATE_CO_RECEIVED"}, inplace = True)
        except Exception as e:
            logger.error("Could not processe data : {}".format(e))
            raise
        logger.info("processing data successfully")

    def handle_missing_values(self):
        try:
            # handle missing value of SUB_PRODUCT
            self.products.set_value(51, 'SUB_PRODUCT', "Credit card")
            self.products.set_value(52, 'SUB_PRODUCT', "Credit reporting")
            self.products.set_value(71, 'SUB_PRODUCT', "Payday loan")
        except Exception as e:
            logger.error("Could not handle missing value : {}".format(e))
            raise
        logger.info("handling missing value successfully")

    def merge(self):
        try:
            # merge data
            self.df = pd.merge(self.complaints_users, self.complaints_companies, on = "COMPLAINT_ID") \
               .merge(self.products, on = "PRODUCT_ID") \
               .merge(self.issues, on = "ISSUE_ID")
            #self.df = self.df.sample(1000)
        except Exception as e:
            logger.error("Could not merge data : {}".format(e))
            raise
        logger.info("merge data successfully")

    def text_cleaning(self):
        try:
            # text cleaning
            self.df[self.opt.text_column] = self.df[self.opt.text_column].apply(lambda x : remove_delimiter(x))
            self.df[self.opt.text_column] = self.df[self.opt.text_column].apply(lambda x : remove_separator(x))
            self.df[self.opt.text_column] = self.df[self.opt.text_column].apply(lambda x : remove_empty(x))
            self.df[self.opt.text_column] = self.df[self.opt.text_column].apply(lambda x : remove_two_spaces(x))
            self.df[self.opt.text_column] = self.df[self.opt.text_column].apply(lambda x : remove_three_spaces(x))
        except Exception as e:
            logger.error("Could not clean data : {}".format(e))
            raise
        logger.info("cleaning data successfully")

    def get_label_mapping(self):
        try:
            self.products["main_append_sub"] = [m+self.concatenate_token+s for m, s in zip(self.products.MAIN_PRODUCT, self.products.SUB_PRODUCT)]
            self.id_to_main_sub = {i:main_sub_str for i, main_sub_str in enumerate(self.products.main_append_sub.unique())}
            self.id_to_main_sub_inv = {main_sub_str:i for i, main_sub_str in self.id_to_main_sub.items()}
        except Exception as e:
            logger.error("Could not get label mapping : {}".format(e))
            raise
        logger.info("get label successfully")

    def mapping(self):
        try:
            self.df["class_label_str"] = [m+self.concatenate_token+s for m, s in zip(self.df.MAIN_PRODUCT, self.df.SUB_PRODUCT)]
            self.df["class_label"] = self.df.class_label_str.apply(lambda x : self.id_to_main_sub_inv[x])
        except Exception as e:
            logger.error("Could not mapping : {}".format(e))
            raise
        logger.info("maximum number of class : {}".format(self.df["class_label"].nunique()))
        logger.info("mapping successfully")

    def label_generation(self):
        """only top frequent N product in dataset will be covered by model"""
        assert self.opt.N <= self.df.class_label.nunique(), logger.error("N should be less or equal than number of unique sub-products : {}".format(self.opt.N))

        self.coverage = self.df.class_label.value_counts(normalize = True).cumsum().iloc[:self.opt.N].iloc[-1] * 100.0

        try:
            self.id_to_main_sub = {i:v for i,v in enumerate(self.df.class_label_str.value_counts().iloc[:self.opt.N].index)}
            if self.opt.N == self.products.main_append_sub.nunique():
                logger.info("# class : {}".format(self.opt.N))
            else:
                self.id_to_main_sub[self.opt.N] = "Others Sub Product"
                logger.info("# class : {}".format(self.opt.N + 1))
            self.id_to_main_sub_inv = {v:i for i,v in self.id_to_main_sub.items()}
            # get label columns
            self.df[self.opt.label_column] = self.df.class_label_str.apply(lambda x : self.id_to_main_sub_inv[x] if x in self.id_to_main_sub_inv else self.id_to_main_sub_inv["Others Sub Product"])
        except Exception as e:
            logger.error("Could not label data : {}".format(e))
            raise
        logger.info("coverage : {}".format(self.coverage))
        logger.info("generating labeled data successfully")

    def save_mapping_sub_to_main(self):
        try:
            if not os.path.exists('../asset'):
                os.mkdir('../asset')
                # Pickling (serializing) a dictionary into a file
            with open('../asset/id_to_main_sub.pickle', 'wb') as filename:
                pickle.dump(self.id_to_main_sub, filename)
        except Exception as e:
            logger.error("Could not save mapping : {}".format(e))
            raise
        logger.info("saving maping dict successfully")

    def split(self):
        """train/test split"""
        assert 0 <= self.opt.train_test_ratio < 1, logger.error("train_test_ratio should between 0~1")
        
        try:
            self.df_train, self.df_test = train_test_split(self.df, test_size = self.opt.train_test_ratio)
        except Exception as e:
            logger.error("Could not do train/test split : {}".format(e))
            raise
        logger.info("train/test split successfully")

    def save(self):
        try:
            # save train/test result
            df_to_txt(self.df_train, os.path.join(self.opt.output_path, "train.txt"), text_column = self.opt.text_column, label_column = self.opt.label_column)
            df_to_txt(self.df_test, os.path.join(self.opt.output_path, "test.txt"), text_column = self.opt.text_column, label_column = self.opt.label_column)
        except Exception as e:
            logger.error("Could not save: {}".format(e))
            raise
        logger.info("save train/test successfully")

    def run(self):
        if self.opt.load_from_db:
            # retrive data from db
            self.retrieve_data_from_db()
        else:
            # load data from local
            self.load_data()
        # process data
        self.data_processing()
        # handle missing value
        self.handle_missing_values()
        # merge
        self.merge()
        # cleaning text
        self.text_cleaning()
        # get mapping of main_sub to id
        self.get_label_mapping()
        # mapping
        self.mapping()
        # get label column
        self.label_generation()
        # save mapping
        self.save_mapping_sub_to_main()
        # train/test split
        self.split()
        # save train/test 
        self.save()

def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../../data', type=str, help = "data path")
    parser.add_argument('--output_path', default='../../data', type=str, help = "path to save train/test text data ")
    parser.add_argument('--train_test_ratio', default=0.2, type=float, help='set ratio between 0 and 1 for train/test split')
    parser.add_argument('--text_column', default="COMPLAINT_TEXT", type=str, help='text column')
    parser.add_argument('--label_column', default="class_label", type=str, help='text column')
    parser.add_argument('--N', default=98, type=int, help='N is most frequent N sub-products that our model will cover.')
    parser.add_argument('--seed', default=1030, type=int, help='set seed for reproducibility and experimentation')
    parser.add_argument('--load_from_db', default=1, type=int, help='load data from db or local file')
    parser.add_argument('--tb_list', default=['complaints_users', 'complaints_companies', 'issues', 'products'], type=list, help='list of table names you want to query from db')

    opt = parser.parse_args()

    # fixed seed if needed
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)

    # logging
    log_file = '../logs/data_process_num_class_{}-{}.log'.format(opt.N, strftime("%y%m%d-%H%M", localtime()))
    if not os.path.exists('../logs'):
        os.mkdir('../logs')
    logger.addHandler(logging.FileHandler(log_file))

    dp = DataProcess(opt)
    dp.run()

if __name__ == "__main__":
    main()