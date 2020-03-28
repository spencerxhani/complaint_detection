import pickle

class Configs:
	
    concatenate_token = "-[CON]-"

    # Unpickling (de-serializing) a dictionary
    with open('../asset/id_to_main_sub.pickle', 'rb') as filename:
        id_to_main_sub = pickle.load(filename)






