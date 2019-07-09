import numpy as np
from atom_distance import atom_distance
from code.feature_enginner.scalar_coupling_contributions import contribution

def merge_features():
    train_X = []
    names = []

    # READ each features
    data, name = atom_distance()
    train_X.append(data)
    names.extend(name)

    data, name = contribution()
    train_X.append(data)
    names.extend(name)

    train_X = np.hstack(train_X)
    return train_X, names

merge_features