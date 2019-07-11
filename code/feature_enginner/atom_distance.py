import pandas as pd

def atom_distance():
    X = pd.read_pickle('../../tmp_data/train_X.pkl')
    return X, ['atom_distance']