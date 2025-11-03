import pickle

def unpickle(pickle_file):
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)
    
def save_with_pickle(file_name, data):
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)