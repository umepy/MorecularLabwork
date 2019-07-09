import pandas as pd

def atom_distance():
    X = pd.read_pickle('data/train_X.pkl')
    return X, ['atom_distance']