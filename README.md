# Challenge - ML Engineer - NLP

> A codebase of end-to-end NLP project: from retriving the data from the DB, EDA, Model Selection/Validation, and 
to Operationalization.

### A project directory layout

    .
    ├── misc                   # database config
    ├── artifacts              # Model persistence directory
    ├── code                   # Source code (alternatively `lib` or `app`)
        ├── asset 
        ├── models             # Deep learning model architecture modules
            ├── bert_document_lstm.py
            ├── bert_scc.py
        ├── src                # Source code
            ├── data_process.py     # data loading and data manipulation
            ├── train.py            # training model
            ├── predict.py          # predict module
            ├── main.py             # to accpet command-line arguments which specify if you want to train a model or make a prediction
            ├── etc..
    ├── data                   # put train/val data
    ├── notebooks              # including .html and .ipynb version of notebook
    ├── requirements.txt       # dependencies
    └── README.md

## Methods

1. Problem Definition: formulate the problem into multi-class classification task

    - 1.1 I append main-product into each sub-product as our unique products(generate classes), 98 products(classes) totally.
        
        - 1.1.1 For example, let's say MAIN_PRODUCT is **Prepaid card**	and SUB_PRODUCT is **Mobile wallet**, then I got one unique product called as one class **Prepaid card-[CON]-Mobile wallet**
        
    - 1.2 Compare top N number of products vs coverage to determine if our model should cover all products.
        - 1.2.1 For example, let's consider **N = 5**, we have around **50% coverage** of all prodcuts
        - 1.2.1 For example, let's consider **N = 50**, we have around **98% coverage** of all prodcuts
    

2. Train/Test split: split dataset int train and one hold-out validatiing set to do experiment(80/20)
    
3. Models

    - 3.1 Fasttext
        - 3.1.1 I use fasttext model as benchmark because it can be done quicky(without gpu needed) and get not bad resully usually.
        - 3.1.2 Please find my experiment result in Question. c.ipynb
    - 3.2 BERT single sentence classification
        - 3.2.1 I used a model considering top 50 most frequen unique product(main+sub product), as known as 51 classes covering around 98% of total products as my final model, saved in artifacts dir.
        - 3.2.1 It has around 63% accuracy and 0.45 f1 score with only 20 epochs
    - 3.3 BERT document + lstm classificaiton
        - 3.3.1 I take long sentence as a document: split long text into a couple of chunks(sentences)
        - 3.3.2 Using LSTM to capure documen representation
        - 3.3.3 there's no time to train on more epochs to see if it helps compared to 3.2
        
Notes:
- For more details, please see my code and jupyter notebook

## Preparation to have a detached DB server
We will be utilizing a PostgreSQL 11 server for storing and retrieving data and Docker for hosting it. To do this, first install Docker: https://docs.docker.com/. 

Once installed, run the following commands in your terminal:
1. To run the server:
`docker run -d --name ht_pg_server -v ht_dbdata:/var/lib/postgresql/data -p 54320:5432 postgres:11`
2. Check the logs to see if it is running:
`docker logs -f ht_pg_server`
3. Create the database:
`docker exec -it ht_pg_server psql -U postgres -c "create database ht_db"`
4. Load data in the database:
`pip install -r misc/requirements.txt`
`python misc/etl.py`


## Instuction

To demonstrate the feature, please run the below command line in order.

Notes:
- Follow the Preparation I mentioned in the above first.
- We suggest to use a virtual enviroment with python 3.7 to execute the below scripts.
- At least one GPU resouce wth Tesla V100 16GB for training


1. Prerequisites

```sh
pip3 install -r requirements.txt
```

2. Train:  pulls the raw data from the DB and generates the fitted model artifact (it should be stored under the artifacts 

```sh
python3 main.py --train True
```

3. Predict

```sh
python3 main.py --predict "My father attempted to clear a debt with a collection agency. They informed him that he could settle the debt for a specific amount. He received a letter stating that if wanted to resolve the issue all he would have to do would pay the settlement amount."
```



### Future work

The thing I think it can be improved if more time povided:

1. More Survey on Long Text Classification: Deu to maximum limit of sequence length 512 of BERT and GPU memory, we need a way to cope with long text of our data.
2. Parameter tuning: Because fine-tuning on BERT model is time-consuming requires more GPU computing resources, I do not do this in this phase. Otherwise, I can try
    - 2.1 More epochs than 50
    - 2.2 Tuning Batch Size and Learning Rate
3. For serving, unit test and integration test, CI/CD pipeline, api deployment, and etc should be considered.