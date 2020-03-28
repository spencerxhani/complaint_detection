# data process
data:
		python3 data_process.py --N 50 --data_path ../../data --output_path ../../data
# train model
train:
		python3 train.py --model_name bert_ssc --polarities_dim 51 --batch_size 32 --max_seq_len 170 --num_epoch 50

# train model
train_b64:
		python3 train.py --model_name bert_ssc --polarities_dim 51 --batch_size 64 --max_seq_len 150 --num_epoch 50

# serve model
serve:
		python3 predict.py --model_name bert_ssc --state_dict_path ../state_dict/bert_ssc_val_acc --polarities_dim 51 --max_seq_len 150

